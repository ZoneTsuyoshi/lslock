import os
import sys
import time
import math
import shutil
import numpy as np
import matplotlib.pyplot as plt 
import configparser

import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torch.backends import cudnn
from torchvision import datasets, transforms
from torch.autograd import Variable

sys.path.append("../src")
from util_functions import mean_squared_error
from convlstm import ConvLSTM


def main():
    model_name = "convlstm"
    system = "ccw" # ccw, gf
    system_dir = {"ccw":"concentric_circle_wave", "gf":"global_flow"}
    result_dir = os.path.join("results", system_dir[system], "convlstm")
    config = configparser.ConfigParser()
    config.read("config_{}.ini".format(system))

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    # shutil.copy("config_{}.ini".format(system), result_dir)
    
    
    ## Set seed and cuda
    seed = 128
    print("Set seed {}.".format(seed))
    
    cuda = torch.cuda.is_available()
    gpu = int(config.get("train", "gpu"))
    if cuda:
        print("cuda is available")
        device = torch.device('cuda', gpu)
    else:
        print("cuda is not available")
        device = torch.device('cpu')
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    # cuda = False
    # device = torch.device('cpu')
    # np.random.seed(seed)
    # torch.autograd.set_detect_anomaly(True)
    
    
    ## Read data and set parameters
    train_data = np.load(config.get("data", "path")).astype(np.float32) # T x h x w
    true_data = np.load(config.get("data", "true_path")).astype(np.float32)
        
    timesteps = int(config.get("data", "timesteps"))
    width = int(config.get("data", "width"))
    height = int(config.get("data", "height"))
    
    loss_name = config.get("network", "loss")
    #n_layers = int(config.get("network", "n_layers"))
    step = int(config.get("network", "step"))
    effective_step = [int(i) for i in config.get("network", "effective_step").split(",")]
    input_channels = int(config.get("network", "input_channels"))
    kernel_size = tuple([int(i) for i in config.get("network", "kernel_size").split(",")])
    n_channels = [int(i) for i in config.get("network", "n_channels").split(",")]
    n_layers = len(n_channels)
    batch_norm = bool(config.get("network", "batch_norm"))
    effective_layers = [int(i) for i in config.get("network", "effective_layers").split(",")]
    
    num_epochs = int(config.get("train", "num_epochs"))
    batch_size = int(config.get("train", "batch_size"))
    optimizer_name = config.get("train", "optimizer")
    init_lr = float(config.get("train", "init_lr"))
    decay_rate = float(config.get("train", "decay_rate"))
    decay_steps = float(config.get("train", "decay_steps"))
    train_steps = int(config.get("train", "train_steps"))
    test_steps = int(config.get("train", "test_steps"))
    prediction_steps = int(config.get("train", "prediction_steps"))
    
    display_steps = int(config.get("logs", "display_steps"))
    save_steps = int(config.get("logs", "save_steps"))

    
    ## Read model
    model = ConvLSTM((height, width), input_channels, n_channels, kernel_size, n_layers, effective_layers, batch_norm, device=device).to(device)
    if loss_name == "MSE":
        loss_fn = nn.MSELoss()
    elif loss_name == "CE":
        loss_fn = nn.CrossEntropyLoss()

    if cuda:
        cudnn.benchmark = True
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    elif optimizer_name == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=init_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_steps, gamma=decay_rate)
    
    
    # Define functions
    def train(epoch):
        model.train()
        epoch_loss = 0
        
        data = Variable(torch.from_numpy(train_data[:train_steps-1])).unsqueeze(1).unsqueeze(1).to(device) # T x bs(=1) x c(=1) x h x w
        # forward + backward + optimize
        optimizer.zero_grad()
        outputs, _ = model(data)
        loss = loss_fn(outputs.squeeze(), torch.from_numpy(train_data[1:train_steps]).to(device))
        loss.backward()
        optimizer.step()
        epoch_loss = loss.item()
                
        if epoch%display_steps==0:
            print_contents = "Train Epoch: [{}/{}]".format(epoch, num_epochs)
            print_contents += "\t {}: {:.6f}".format(
                                loss_name,
                                epoch_loss)
            print(print_contents)
        return epoch_loss
    
    
    def test(epoch):
        """uses test data to evaluate likelihood of the model"""
        model.eval()
        epoch_loss = 0

        data = Variable(torch.from_numpy(train_data[train_steps:train_steps+test_steps-1])).unsqueeze(1).unsqueeze(1).to(device) # T x bs(=1) x c(=1) x w x h
        # forward + backward + optimize
        optimizer.zero_grad()
        outputs, _ = model(data)
        loss = loss_fn(outputs.squeeze(), torch.from_numpy(train_data[train_steps+1:train_steps+test_steps]).to(device))
        loss.backward()
        optimizer.step()
        epoch_loss = loss.item()
            
        if epoch%display_steps==0:
            print_contents = "====> Test set loss:"
            print_contents += " {} = {:.4f}".format(loss_name,
                                                    epoch_loss)
            print(print_contents)
        return epoch_loss
        
    
    def prediction(epoch):
        """n-step prediction"""
        model.eval()
        loss = np.zeros((2, prediction_steps))
        output = np.zeros((prediction_steps, train_data.shape[1], train_data.shape[2]))
        
        data = Variable(torch.from_numpy(train_data[:train_steps-1].squeeze()))
        data = data.unsqueeze(1).unsqueeze(1).to(device) # T x bs(=1) x c(=1) x h x w
        outputs, last_state_list = model(data)
        #prev_state = outputs[-1].view(1,1,1,height,width) # T(=1) x bs(=1) x c(=1) x h x w
        prev_state = Variable(torch.from_numpy(train_data[train_steps])).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
        
        for i in range(prediction_steps):
            prev_state, last_state_list = model(prev_state, last_state_list)
            loss[0,i] = mean_squared_error(prev_state.squeeze().cpu().detach().numpy(), train_data[train_steps+i])
            loss[1,i] = mean_squared_error(prev_state.squeeze().cpu().detach().numpy(), true_data[train_steps+i])
            output[i] = prev_state.squeeze().cpu().detach().numpy()
        
        if epoch%display_steps==0:
            print_contents = "===> Prediction loss:\n"
            for i in range(prediction_steps):
                print_contents += "{} step forecast {}: {}\n".format(i+1, loss_name, loss[0,i])
            print(print_contents)
        
        #print("output", output.shape, output.min(), output.max())
        return loss, output
        
    
    ## Train model
    def execute():
        train_loss = np.zeros(num_epochs)
        test_loss = np.zeros(num_epochs)
        prediction_loss = np.zeros((num_epochs, 2, prediction_steps))
        outputs = np.zeros((num_epochs//save_steps, prediction_steps, train_data.shape[1], train_data.shape[2]))
        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            # training + testing
            _train_loss = train(epoch)
            _test_loss = test(epoch)
            _prediction_loss = prediction(epoch)
            scheduler.step()

            # substitute losses for array
            train_loss[epoch-1] = _train_loss
            test_loss[epoch-1] = _test_loss
            prediction_loss[epoch-1], outputs[(epoch-1)//save_steps] = _prediction_loss

            # duration
            duration = int(time.time() - start_time)
            second = int(duration%60)
            remain = int(duration//60)
            minute = int(remain%60)
            hour = int(remain//60)
            print("Duration: {} hour, {} min, {} sec.".format(hour, minute, second))
            remain = (num_epochs - epoch) * duration / epoch
            second = int(remain%60)
            remain = int(remain//60)
            minute = int(remain%60)
            hour = int(remain//60)
            print("Estimated Remain Time: {} hour, {} min, {} sec.".format(hour, 
                                                                           minute, 
                                                                           second))

            # saving model
            if epoch % save_steps == 0:
                torch.save(model.state_dict(), os.path.join(result_dir, 
                                        'state_dict_'+str(epoch)+'.pth'))
                torch.save(optimizer.state_dict(), os.path.join(result_dir,
                                        'adam_state_dict_'+str(epoch)+'.pth'))
                print('Saved model to state_dict_'+str(epoch)+'.pth')
                # np.save(os.path.join(result_dir, "train_loss.npy"), train_loss)
                # np.save(os.path.join(result_dir, "test_loss.npy"), test_loss)
                np.save(os.path.join(result_dir, "prediction_loss.npy"), prediction_loss)
                np.save(os.path.join(result_dir, "convlstm_mse.npy"), prediction_loss[:,1])
                # np.save(os.path.join(result_dir, "prediction.npy"), outputs)

                # plot loss for train and test
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                ax.plot(range(epoch), train_loss[:epoch],
                              label="train")
                ax.plot(range(epoch), test_loss[:epoch],
                              label="test")
                ax.set_xlabel("epoch")
                ax.set_ylabel(loss_name)
                ax.legend()
                fig.savefig(os.path.join(result_dir, "loss.png"), 
                            bbox_inches="tight")
                ax.set_yscale("log")
                fig.savefig(os.path.join(result_dir, "log_loss.png"), 
                            bbox_inches="tight")
                
                # plot prediction loss
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                for i in range(save_steps, epoch+1, save_steps):
                    ax.plot(range(train_steps, train_steps+prediction_steps), prediction_loss[i-1,0,:],
                                          label="epoch={}".format(i))
                ax.set_xlabel("timestep")
                ax.set_ylabel(loss_name)
                ax.legend()
                fig.savefig(os.path.join(result_dir, "prediction_loss.png"), 
                            bbox_inches="tight")
                ax.set_yscale("log")
                fig.savefig(os.path.join(result_dir, "log_prediction_loss.png"), 
                            bbox_inches="tight")
                
    
    
    ## Excute
    # measure for culabas rutine error
    execute()

    
if __name__ == "__main__":
    main()