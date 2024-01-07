from json import load

from libs.plot import plot_values


def plot():
    """
    [
      10.979412078857422, 6.658339500427246, 6.300214767456055, 6.093669891357422,
      5.90362024307251, 5.726109504699707, 5.6165924072265625, 5.497467041015625,
      5.342617988586426, 5.261222839355469, 5.256556987762451, 5.224802017211914,
      5.10974645614624, 5.00878381729126, 5.053899765014648, 5.006225109100342,
      4.8698649406433105, 4.960547924041748, 4.763721466064453, 4.775387763977051,
      4.790281772613525, 4.616654396057129, 4.736024856567383, 4.677309513092041,
      4.712619781494141, 4.6578192710876465, 4.719181060791016, 4.566228866577148,
      4.492823600769043, 4.506204128265381, 4.528460502624512, 4.483500003814697,
      4.514330863952637, 4.474279880523682, 4.512044429779053, 4.381807804107666,
      4.483075141906738, 4.345661640167236, 4.355191230773926, 4.403414249420166,
      4.429179668426514, 4.3423686027526855, 4.321564197540283, 4.335482597351074,
      4.308663845062256, 4.291524410247803, 4.376142978668213, 4.359085559844971,
      4.332291603088379, 4.297501087188721, 4.245373725891113, 4.193171501159668,
      4.131768703460693, 4.26424503326416, 4.1956400871276855, 4.227156639099121,
      4.2483930587768555, 4.2457709312438965, 4.193092346191406, 4.13712739944458,
      4.0710062980651855, 4.093221664428711, 4.091123580932617, 4.147737503051758,
      4.199992656707764, 4.162751197814941, 4.212997913360596, 4.194666385650635,
      4.215516567230225, 4.102850437164307, 4.076639652252197, 4.110166072845459,
      4.028319358825684, 4.14346170425415, 4.126922607421875, 4.0074992179870605,
      4.08997106552124, 4.082268238067627, 4.060946941375732, 4.073566436767578,
      4.040389537811279, 4.060932636260986, 4.085163116455078, 4.087076187133789,
      4.081695556640625, 3.9936413764953613, 4.062099933624268, 4.138188362121582,
      3.951782464981079, 4.00286340713501, 4.073570728302002, 4.006319522857666,
      4.049221992492676, 3.95086932182312, 4.045578956604004, 4.0174946784973145,
      3.967092514038086, 4.112637519836426, 4.0414886474609375, 3.9588966369628906,
      3.875328779220581, 3.91062593460083, 4.015019416809082, 4.039207458496094,
      3.984792709350586, 3.989450216293335, 3.9802064895629883, 4.041986465454102,
      4.013123989105225, 3.861638307571411, 3.914746046066284, 3.8556885719299316,
      4.000454902648926, 3.8862462043762207, 3.956770896911621, 3.9131994247436523,
      3.925379991531372, 3.917194366455078, 3.913381814956665, 4.004879474639893,
      3.936905860900879, 3.9344048500061035, 3.900839328765869, 3.899947166442871,
      3.9214730262756348, 4.023489475250244, 3.999176025390625, 3.8513317108154297,
      3.8790366649627686, 3.8092334270477295, 3.799619436264038, 3.917860746383667,
      3.928176164627075, 3.898284435272217, 3.8477416038513184, 3.960604429244995,
      4.007314205169678, 3.88840651512146, 3.8642923831939697, 3.9649267196655273,
      3.928189277648926, 3.8568499088287354, 3.865957736968994, 3.8495335578918457,
      3.9089534282684326, 3.855912685394287, 3.829118490219116, 3.9562621116638184,
      3.8340907096862793, 3.889315366744995, 3.8504550457000732, 3.8964622020721436,
      3.81813383102417, 3.7849690914154053, 3.8646352291107178, 3.858227014541626,
      3.899744987487793, 3.7882723808288574, 3.8877975940704346, 3.773981809616089,
      3.845196485519409, 3.9577622413635254, 3.8410470485687256, 3.792508125305176,
      3.771611452102661, 3.8438057899475098, 3.834071159362793, 3.7823991775512695,
      3.8100690841674805, 3.8020846843719482, 3.8275773525238037,
      3.9041121006011963, 3.870351791381836, 3.899260997772217, 3.885814666748047,
      3.896026134490967, 3.8609583377838135, 3.8680496215820312, 3.8971288204193115,
      3.8993165493011475, 3.8800947666168213, 3.7712137699127197,
      3.8463006019592285, 3.766627788543701, 3.8928794860839844, 3.8197460174560547,
      3.6398792266845703, 3.7050116062164307, 3.8103041648864746,
      3.7878313064575195, 3.792742967605591, 3.7712337970733643, 3.7732741832733154,
      3.824061155319214, 3.7612998485565186, 3.8111753463745117, 3.8297088146209717,
      3.8144118785858154
    ]
    """
    losses: list[float] = []

    with open("week4/data/loss.v2.json", "r") as f:
        losses += load(f)

    plot_values(train=losses)


if __name__ == "__main__":
    plot()
