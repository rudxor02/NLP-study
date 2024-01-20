# reference: https://github.com/xiaojunxu/SQLNet/blob/master/sqlnet/utils.py#


import json

from datasets import load_dataset
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaModel, LlamaTokenizer

from week6.config import config
from week6.dbengine import DBEngine
from week6.table_linearize import IndexedRowTableLinearize


def load_data(sql_paths, table_paths, use_small=False):
    if not isinstance(sql_paths, list):
        sql_paths = (sql_paths,)
    if not isinstance(table_paths, list):
        table_paths = (table_paths,)
    sql_data = []
    table_data = {}

    max_col_num = 0
    for SQL_PATH in sql_paths:
        print(f"Loading data from {SQL_PATH}")
        with open(SQL_PATH) as inf:
            for idx, line in enumerate(inf):
                if use_small and idx >= 1000:
                    break
                sql = json.loads(line.strip())
                sql_data.append(sql)

    for TABLE_PATH in table_paths:
        print(f"Loading data from {TABLE_PATH}")
        with open(TABLE_PATH) as inf:
            for line in inf:
                tab = json.loads(line.strip())
                table_data[tab["id"]] = tab

    for sql in sql_data:
        assert sql["table_id"] in table_data

    return sql_data, table_data


def load_dataset(dataset_id, use_small=False):
    if dataset_id == 0:
        print("Loading from original dataset")
        sql_data, table_data = load_data(
            "week6/data/train_tok.jsonl",
            "week6/data/train_tok.tables.jsonl",
            use_small=use_small,
        )
        val_sql_data, val_table_data = load_data(
            "week6/data/dev_tok.jsonl",
            "week6/data/dev_tok.tables.jsonl",
            use_small=use_small,
        )
        test_sql_data, test_table_data = load_data(
            "week6/data/test_tok.jsonl",
            "week6/data/test_tok.tables.jsonl",
            use_small=use_small,
        )
        TRAIN_DB = "week6/data/train.db"
        DEV_DB = "week6/data/dev.db"
        TEST_DB = "week6/data/test.db"
    else:
        print("Loading from re-split dataset")
        sql_data, table_data = load_data(
            "week6/data_resplit/train.jsonl",
            "week6/data_resplit/tables.jsonl",
            use_small=use_small,
        )
        val_sql_data, val_table_data = load_data(
            "week6/data_resplit/dev.jsonl",
            "week6/data_resplit/tables.jsonl",
            use_small=use_small,
        )
        test_sql_data, test_table_data = load_data(
            "week6/data_resplit/test.jsonl",
            "week6/data_resplit/tables.jsonl",
            use_small=use_small,
        )
        TRAIN_DB = "week6/data_resplit/table.db"
        DEV_DB = "week6/data_resplit/table.db"
        TEST_DB = "week6/data_resplit/table.db"

    return (
        sql_data,
        table_data,
        val_sql_data,
        val_table_data,
        test_sql_data,
        test_table_data,
        TRAIN_DB,
        DEV_DB,
        TEST_DB,
    )


def to_batch_seq(sql_data, table_data, idxes, st, ed, ret_vis_data=False):
    q_seq = []
    col_seq = []
    col_num = []
    ans_seq = []
    query_seq = []
    gt_cond_seq = []
    vis_seq = []
    for i in range(st, ed):
        sql = sql_data[idxes[i]]
        q_seq.append(sql["question_tok"])
        col_seq.append(table_data[sql["table_id"]]["header_tok"])
        col_num.append(len(table_data[sql["table_id"]]["header"]))
        ans_seq.append(
            (
                sql["sql"]["agg"],
                sql["sql"]["sel"],
                len(sql["sql"]["conds"]),
                tuple(x[0] for x in sql["sql"]["conds"]),
                tuple(x[1] for x in sql["sql"]["conds"]),
            )
        )
        query_seq.append(sql["query_tok"])
        gt_cond_seq.append(sql["sql"]["conds"])
        vis_seq.append(
            (sql["question"], table_data[sql["table_id"]]["header"], sql["query"])
        )
    if ret_vis_data:
        return q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, vis_seq
    else:
        return q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq


def to_batch_query(sql_data, idxes, st, ed):
    query_gt = []
    table_ids = []
    for i in range(st, ed):
        query_gt.append(sql_data[idxes[i]]["sql"])
        table_ids.append(sql_data[idxes[i]]["table_id"])
    return query_gt, table_ids


def epoch_exec_acc(
    model: nn.Module, batch_size: int, sql_data, table_data, db_path: str
):
    engine = DBEngine(db_path)

    model.eval()
    perm = list(range(len(sql_data)))
    tot_acc_num = 0.0
    acc_of_log = 0.0
    st = 0
    while st < len(sql_data):
        ed = st + batch_size if st + batch_size < len(perm) else len(perm)
        (
            q_seq,
            col_seq,
            col_num,
            ans_seq,
            query_seq,
            gt_cond_seq,
            raw_data,
        ) = to_batch_seq(sql_data, table_data, perm, st, ed, ret_vis_data=True)
        raw_q_seq = [x[0] for x in raw_data]
        raw_col_seq = [x[1] for x in raw_data]
        gt_where_seq = model.generate_gt_where_seq(q_seq, col_seq, query_seq)
        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        gt_sel_seq = [x[1] for x in ans_seq]
        score = model.forward(
            q_seq, col_seq, col_num, (True, True, True), gt_sel=gt_sel_seq
        )
        pred_queries = model.gen_query(
            score, q_seq, col_seq, raw_q_seq, raw_col_seq, (True, True, True)
        )

        for idx, (sql_gt, sql_pred, tid) in enumerate(
            zip(query_gt, pred_queries, table_ids)
        ):
            ret_gt = engine.execute(tid, sql_gt["sel"], sql_gt["agg"], sql_gt["conds"])
            try:
                ret_pred = engine.execute(
                    tid, sql_pred["sel"], sql_pred["agg"], sql_pred["conds"]
                )
            except:
                ret_pred = None
            tot_acc_num += ret_gt == ret_pred

        st = ed

    return tot_acc_num / len(sql_data)


def epoch_acc(model: nn.Module, batch_size: int, sql_data, table_data, pred_entry):
    model.eval()
    perm = list(range(len(sql_data)))
    st = 0
    one_acc_num = 0.0
    tot_acc_num = 0.0
    while st < len(sql_data):
        ed = st + batch_size if st + batch_size < len(perm) else len(perm)

        (
            q_seq,
            col_seq,
            col_num,
            ans_seq,
            query_seq,
            gt_cond_seq,
            raw_data,
        ) = to_batch_seq(sql_data, table_data, perm, st, ed, ret_vis_data=True)
        raw_q_seq = [x[0] for x in raw_data]
        raw_col_seq = [x[1] for x in raw_data]
        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        gt_sel_seq = [x[1] for x in ans_seq]
        score = model.forward(q_seq, col_seq, col_num, pred_entry, gt_sel=gt_sel_seq)
        pred_queries = model.gen_query(
            score, q_seq, col_seq, raw_q_seq, raw_col_seq, pred_entry
        )
        one_err, tot_err = model.check_acc(raw_data, pred_queries, query_gt, pred_entry)

        one_acc_num += ed - st - one_err
        tot_acc_num += ed - st - tot_err

        st = ed
    return tot_acc_num / len(sql_data), one_acc_num / len(sql_data)


def load_tokenizer() -> LlamaTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        "./week6/data/sft_v1/checkpoint-3523/", local_files_only=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    print("tokenizer loaded")
    return tokenizer


def load_model() -> LlamaModel:
    model = AutoModelForCausalLM.from_pretrained(
        "./week6/data/sft_v1/checkpoint-3523/",
        local_files_only=True,
        cache_dir="/data/hub",
    )
    print("model loaded")
    return model


if __name__ == "__main__":
    # (
    #     sql_data,
    #     table_data,
    #     val_sql_data,
    #     val_table_data,
    #     test_sql_data,
    #     test_table_data,
    #     TRAIN_DB,
    #     DEV_DB,
    #     TEST_DB,
    # ) = load_dataset(0, use_small=False)

    tokenizer = load_tokenizer()
    model = load_model()

    model.to("cuda:5")
    model.eval()

    # predict from prompt
    linearizer = IndexedRowTableLinearize()
    example = {}
    example["table"] = {
        "header": [
            "Player",
            "No.",
            "Nationality",
            "Position",
            "Years in Toronto",
            "School/Club Team",
        ],
        "page_title": "Toronto Raptors all-time roster",
        "page_id": "",
        "types": ["text", "text", "text", "text", "text", "text"],
        "id": "1-10015132-16",
        "section_title": "R",
        "caption": "R",
        "rows": [
            [
                "Aleksandar Radojević",
                "25",
                "Serbia",
                "Center",
                "1999-2000",
                "Barton CC (KS)",
            ],
            [
                "Shawn Respert",
                "31",
                "United States",
                "Guard",
                "1997-98",
                "Michigan State",
            ],
            [
                "Quentin Richardson",
                "N/A",
                "United States",
                "Forward",
                "2013-present",
                "DePaul",
            ],
            [
                "Alvin Robertson",
                "7, 21",
                "United States",
                "Guard",
                "1995-96",
                "Arkansas",
            ],
            [
                "Carlos Rogers",
                "33, 34",
                "United States",
                "Forward-Center",
                "1995-98",
                "Tennessee State",
            ],
            ["Roy Rogers", "9", "United States", "Forward", "1998", "Alabama"],
            [
                "Jalen Rose",
                "5",
                "United States",
                "Guard-Forward",
                "2003-06",
                "Michigan",
            ],
            [
                "Terrence Ross",
                "31",
                "United States",
                "Guard",
                "2012-present",
                "Washington",
            ],
        ],
        "name": "table_10015132_16",
    }
    table_dict = {
        "header": example["table"]["header"],
        "rows": example["table"]["rows"][: config.table_max_rows],
    }
    linear_table = linearizer.process_table(table_dict)
    prompt = (
        "### Table\n"
        + linear_table
        + "\n### Question\n"
        + "What is terrence ross' nationality"
        + "\n### SQL\n"
    )
    prompt_encodings = tokenizer(prompt, return_tensors="pt")
    prompt_encodings = prompt_encodings.to("cuda:5")
    print("predicting...")
    prompt_output = model.generate(
        **prompt_encodings,
        max_length=200,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        num_return_sequences=5,
        temperature=0.9,
    )
    prompt_output = tokenizer.batch_decode(prompt_output)
    print(prompt_output[0])
