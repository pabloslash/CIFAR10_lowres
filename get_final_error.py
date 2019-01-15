# Get deterministic accuracy
import os

############################## DropOut ##############################

model_names = ['cifar10_dropout_01.pt', 'cifar10_dropout_02.pt', 'cifar10_dropout_03.pt',
                'cifar10_dropout_04.pt', 'cifar10_dropout_05.pt']

net = Net_cifar()
net.cuda()

final_error = []

for model in xrange(0, len(model_names)):

    #LOAD
    load_dir = os.getcwd() + '/networks/networks_dropout/' + model_names[model]
    net.load_state_dict(torch.load(load_dir))

    final_error.append(get_accuracy(testloader, net, classes))

fin_dropout_error = np.mean(final_error)
fin_dropout_std = np.std(final_error)

with open('networks/networks_dropout/results/final_error_dropout.pkl', 'w') as f:  # Python 3: open(..., 'wb')
    pickle.dump([fin_dropout_error, fin_dropout_std], f)



############################## NO dropOut ##############################

model_names = ['cifar10_NOdropout_01.pt', 'cifar10_NOdropout_02.pt', 'cifar10_NOdropout_03.pt', 'cifar10_NOdropout_04.pt', 'cifar10_NOdropout_05.pt']

# model_name = 'cifar10_NOdropout_05.pt'

final_error = []

net = Net_cifar()
net.cuda()

for model in xrange(0, len(model_names)):
    print (model_names[model])

    #LOAD
    load_dir = os.getcwd() + '/networks/networks_NOdropout/' + model_names[model]
    net.load_state_dict(torch.load(load_dir))

    final_error.append(get_accuracy(testloader, net, classes))

fin_NOdropout_error = np.mean(final_error)
fin_NOdropout_std = np.std(final_error)

with open('networks/networks_NOdropout/results/final_error_NOdropout.pkl', 'w') as f:  # Python 3: open(..., 'wb')
    pickle.dump([fin_NOdropout_error, fin_NOdropout_std], f)
