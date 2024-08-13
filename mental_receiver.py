import threading
from cortex import Cortex

class MentalReceiver():


    def __init__(self,commands):
        self.c = Cortex(user, debug_mode=False)
        self.c.bind(new_com_data=self.on_new_data)
        self.c.do_prepare_steps()
        self.commands = commands
        self.command = None



    def load_profile(self, profile_name):
        profiles = self.c.query_profile()

        if profile_name not in profiles:
            status = 'create'
            self.c.setup_profile(profile_name, status)

        status = 'load'
        self.c.setup_profile(profile_name, status)


    def unload_profile(self, profile_name):
        profiles = self.c.query_profile()

        if profile_name in profiles:
            status = 'unload'
            self.c.setup_profile(profile_name, status)
        else:
            print("The profile " + profile_name + " is not existed.")


    def train_mc(self, profile_name, training_action, number_of_train):
        print('begin train -----------------------------------')
        num_train = 0
        while num_train < number_of_train:
            num_train = num_train + 1

            print('start training {0} time {1} ---------------'.format(training_action, num_train))
            print('\n')
            status='start'          
            self.c.train_request(detection='mentalCommand',
                                action=training_action,
                                status=status)

            print('accept {0} time {1} ---------------'.format(training_action, num_train))
            print('\n')
            status='accept'
            self.c.train_request(detection='mentalCommand',
                                action=training_action, 
                                status=status)
        
        print('save trained action')
        status = "save"
        self.c.setup_profile(profile_name, status)




    def start(self,profile_name):
        # self.c.setup_profile(profile_name,"load")
        # self.train_mc(profile_name,"push",2)
        # self.c.setup_profile(profile_name,"unload")
        x = threading.Thread(target=self.c.sub_request,args=(["com"],))
        x.start()
        


    def on_new_data(self, *args, **kwargs):
        action = kwargs.get("data")["action"]
        if action in self.commands:
            self.command = action
            print(action)
        else:
            self.command = None
        





user = {
    "client_id" : "KD7TAk1pdrSLO42V3XrYJhxv6d15zgmyRsVwRT98",
    "client_secret" : "vwBoZJiw0fWgKHMweEBRRDkRjGcfk9TK0X95lUcgOuEd2enAqKnYt9T9nsfdaDOtUwd5nie1XFw1wjuQtF7HzATBRbB09L35yHcHRWhAVAFd318ZNrtN7yfEU4sjm6nX",
    "debit" : 100
}