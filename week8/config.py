from pydantic import BaseModel


class Config(BaseModel):
    device: str = "cuda:5"
    num_layers: int = 40
    antonyms_data_path: str = "week8/data/antonyms.json"
    antonyms_hypothesis_result_path: str = "week8/data/results/antonyms.json"
    en_es_data_path: str = "week8/data/en_es.json"
    en_es_hypothesis_result_path: str = "week8/data/results/en_es.json"
    location_country_data_path: str = "week8/data/location_country.json"
    location_country_hypothesis_result_path: str = (
        "week8/data/results/location_country.json"
    )


config = Config()
