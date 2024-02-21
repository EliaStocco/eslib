# from copy import copy
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import warnings
import os
import time
from copy import copy
from eslib.functions import add_default
from eslib.functions import remove_empty_folder
from eslib.functions import remove_files_in_folder
from eslib.nn.plot   import plot_learning_curves
from eslib.nn.training.functions import save_checkpoint
from eslib.nn.training.functions import get_all_dataloader
from eslib.nn.training.functions import get_all

__all__ = ["train"]

yval_real = None
all_dataloader_val = None

ytrain_real = None
all_dataloader_train = None

#----------------------------------------------------------------#
# Attention!
# There is a bit of confusion between 'parameters' and 'parameters':
# some variables are stored in the former and not in the latter (or viceversa) just for simplicity.
#----------------------------------------------------------------#

def train(
    model: torch.nn.Module,
    train_dataset: list,
    val_dataset: list,
    parameters: dict,
    make_dataloader: callable = None,
    opts=None,
):
    """
    Train a neural network model.

    Args:
        model (torch.nn.Module):
            The neural network model to be trained.

        train_dataset (list):
            List of training data.

        val_dataset (list):
            List of validation data.

        parameters (dict):
            Dictionary containing various parameters for training.

        make_dataloader (callable, optional):
            A function that creates a data loader from a dataset. Defaults to None.

        opts (dict, optional):
            Additional options for training. Defaults to None.

    Returns:
        tuple:
            A tuple containing the trained model, dataframe, correlation, and information about the training.
    """

    ##########################################
    # time
    start_task_time = time.time()

    print("\n\tTraining\n")
    # print("\tPreparing training")

    # information about the status of the training
    info = "all good"

    ##########################################
    default = {
        "plot" : {
            "learning-curve": {
                "N": 10
            }
        },
        "dataloader" : {
            "shuffle": False
        },
        "thr" : {
            "exit": 10000
        },
        "disable"        : False,
        "restart"        : False,
        "recompute_loss" : False,
        "save"           : {
            "parameters" : 1
        },
    }
    opts = add_default(opts, default)

    ##########################################
    # set the necessary default parameters
    default = {
        "scheduler" : "",   # None
        "name"      : None, # it will be set later to "untitled", but a warning will be raised
        "output"    : None, # it will be set later to "results",  but a warning will be raised
        "bs"        : -1,
        "n_epochs"  : 100,
        "optimizer" : "adam",
        "loss"      : "mse",
        "lr"        : 1e-2,
    }
    parameters = add_default(parameters, default)

    print("\tHyperparameters:")
    for k in parameters.keys():
        try :
            print("\t\t{:20s}: ".format(k),parameters[k])
        except:
            if k == "loss":
                if type(parameters["loss"]) == str:
                    print("\tloss: ", parameters["loss"])
            else:
                pass

    ##########################################
    # default values
    if make_dataloader is None:
        from eslib.nn.dataset import make_dataloader as _make_dataloader
        make_dataloader = \
            lambda dataset, batch_size, shuffle=opts["dataloader"]["shuffle"]: \
                _make_dataloader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

    # output folder
    if parameters["output"] is None:
        warnings.warn("'output' is None: specify a folder name to print some information to file.\n\
                      'results' will be set as default")
        parameters["output"] = "results"

    # the name of the output files
    if parameters["name"] is None:
        warnings.warn("'name' is None: specify a filename to distinguish the results from other parameters")
        parameters["name"] = "untitled"

    ##########################################
    # output folders
    folders = {
        "parameters"  : "{:s}/parameters".  format(parameters["output"]),
        "dataframes"  : "{:s}/dataframes".  format(parameters["output"]),
        "images"      : "{:s}/images".      format(parameters["output"]),
        "correlations": "{:s}/correlations".format(parameters["output"]),
    }

    # create the output folders
    for folder in [parameters["output"], *folders.values()]:
        if not os.path.exists(folder):
            print("\tCreating folder '{:s}'".format(folder))
            os.mkdir(folder)

    ##########################################
    # set the loss function
    if type(parameters["loss"]) == str:
        if parameters["loss"].lower() == "mse":
            from torch.nn import MSELoss
            loss_fn = MSELoss()
        else:
            raise ValueError("loss function not known")
    else :
        loss_fn = parameters["loss"]
        print("\n\tprovided loss function will be supposed to be 'callable'")

    ##########################################
    # set the optimizer
    match parameters["optimizer"].lower():
        case "adam":
            from torch.optim import Adam
            optimizer = Adam(   params=model.parameters(), lr=parameters["lr"])
        case "adamw":
            from torch.optim import AdamW
            optimizer = AdamW(  params=model.parameters(), 
                                lr=parameters["lr"],
                                weight_decay=parameters["weight_decay"])
        case _:
            raise ValueError("optimizer not known")
            
    ##########################################
    # set the learning rate scheduler
    movingaverage = None
    scheduler = None
    movingaverageLR = None
    tmp = parameters["scheduler"].lower()
    if tmp == "" :
        scheduler = None
    elif tmp == "none" :
        scheduler = None
    elif tmp == "plateau":
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(optimizer,factor=parameters["scheduler-factor"],patience=parameters["scheduler-patience"])
    
        # case "ma":
        #     from eslib.nn.functions import MovingAverage
        #     movingaverage = MovingAverage(parameters["scheduler-window"])
        #     movingaverageLR = MovingAverage(parameters["scheduler-window"])
        #     from torch.optim.lr_scheduler import LambdaLR
        #     def myscheduler(epoch):
        #         if epoch < parameters["scheduler-epoch"]:
        #             movingaverageLR.update(1)
        #             return movingaverageLR.get_ma() 
        #         else:
        #             new = 1 - parameters["scheduler-max"] * np.exp( - movingaverage.get_madt() )
        #             movingaverageLR.update(new)
        #             return movingaverageLR.get_ma()
        #     scheduler = LambdaLR(optimizer,lr_lambda=myscheduler)
    else:
        raise ValueError("scheduler not known")
        
    ##########################################
    # prepare the dataloaders for the train and validation datasets
    batch_size = int(parameters["bs"])
    n_epochs = int(parameters["n_epochs"])
    dataloader_train = make_dataloader(train_dataset, batch_size)
    batches_per_epoch = len(dataloader_train)
    train_loss_one_epoch = np.full(batches_per_epoch, np.nan)

    # give a summary of the length of the following for cycles
    print("\n\tSummary:")
    print("\t      n. of epochs:", n_epochs)
    print("\t        batch size:", batch_size)
    print("\t  n. of iterations:", batches_per_epoch)
    print("\ttrain dataset size:", len(train_dataset))
    print("\t  val dataset size:", len(val_dataset))
    print("\n")

    ##########################################
    # dataframe
    dataframe = pd.DataFrame(
        np.nan,
        columns=["epoch", "train", "val", "std", "ratio","lr"],
        index=np.arange(n_epochs),
    )
    if opts["recompute_loss"]:
        dataframe["train-2"] = None
        dataframe["ratio-2"] = None
    else:
        dataframe["ratio"] = None

    ##########################################
    # compute the real values of the validation dataset only once
    global yval_real, all_dataloader_val
    yval_real = None
    if yval_real is None or not opts["keep_dataset"]:
        print("\tcompute validation dataset output (this will save time in the future)")
        argv = {
            "dataset": val_dataset,
            "make_dataloader": make_dataloader,
        }
        yval_real = get_all(model, **argv)
        all_dataloader_val = get_all_dataloader(**argv)

    global ytrain_real, all_dataloader_train
    ytrain_real = None
    if ytrain_real is None or not opts["keep_dataset"] and opts["recompute_loss"]:
        print("\tcompute training dataset output (this will save time in the future)")
        argv = {
            "dataset": train_dataset,
            "make_dataloader": make_dataloader,
        }
        ytrain_real = get_all(model, **argv)
        all_dataloader_train = get_all_dataloader(**argv)

    ##########################################
    # prepare output files
    savefiles = {
        "dataframes": "{:s}/{:s}.csv".format(folders["dataframes"], parameters["name"]),
        "images": "{:s}/{:s}.pdf".format(folders["images"], parameters["name"]),
    }

    ##########################################
    # prepare checkpoint
    # https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html

    start_epoch = 0

    checkpoint_folder = "checkpoint"
    if not os.path.exists(checkpoint_folder):
        os.mkdir(checkpoint_folder)

    parameters_folder = "{:s}/{:s}".format(folders["parameters"], parameters["name"])
    if not os.path.exists(parameters_folder):
        os.mkdir(parameters_folder)
    elif parameters["restart"]:
        print("\tCleaning parameters folder '{:s}'".format(parameters_folder))
        remove_files_in_folder(parameters_folder, "pth")

    checkpoint_file = "{:s}/{:s}.pth".format(checkpoint_folder, parameters["name"])
    if os.path.exists(checkpoint_file) and not opts["restart"]:
        print("\tReading checkpoint from file '{:s}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)

        start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model_state_dict"],strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None :
            if 'scheduler_state_dict' not in checkpoint:
                warnings.warn("No 'scheduler_state_dict' found in checkpoint file")
            else :
                try :
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except:
                    warnings.warn("Problem reading 'scheduler_state_dict' from checkpoint file")
        if movingaverage is not None :
            if 'ma' not in checkpoint:
                warnings.warn("No 'ma' found in checkpoint file")
            else :
                try :
                    movingaverage.load_state_dict(checkpoint['ma'])
                except:
                    warnings.warn("Problem reading 'ma' from checkpoint file")
        if movingaverageLR is not None :
            if 'maLR' not in checkpoint:
                warnings.warn("No 'maLR' found in checkpoint file")
            else :
                try :
                    movingaverageLR.load_state_dict(checkpoint['maLR'])
                except:
                    warnings.warn("Problem reading 'maLR' from checkpoint file")

        # read array
        if os.path.exists(savefiles["dataframes"]):
            tmp = pd.read_csv(savefiles["dataframes"])
            if not opts["recompute_loss"] and "train-2" in tmp:
                dataframe["train-2"] = None
                dataframe["ratio-2"] = None
            dataframe.iloc[: len(tmp)] = copy(tmp)
            del tmp

        if opts["recompute_loss"] and "train-2" not in dataframe:
            dataframe["train-2"] = None
            dataframe["ratio-2"] = None
    elif not os.path.exists(checkpoint_file):
        print("\tno checkpoint file found")
    elif opts["restart"]:
        print("\trestart=True")

    ##########################################
    print("\n\tOptimizer parameters")
    for k,i in optimizer.param_groups[0].items():
        if k == 'params' : 
            continue
        print("\t{:>15s}:".format(k),i)

    ##########################################
    # start the training procedure
    print("\n\t...and here we go!\n")
    k = copy(start_epoch)
    for epoch in range(start_epoch, n_epochs):

        if info != "all good":
            break

        ##########################################
        # measure elapsed time
        now = time.time()
        if (
            abs(now - opts["start_time"] + 60) > parameters["max_time"]
            and parameters["max_time"] > 0
        ):
            print("\n\tMaximum time reached: stopping")
            info = "time over"
            break
        if (
            abs(now - start_task_time + 60) > parameters["task_time"]
            and parameters["task_time"] > 0
        ):
            print("\n\tMaximum time per task reached: stopping task")
            info = "time over"
            break

        if os.path.exists("EXIT"):
            info = "exit file detected"
            print("\n\t'EXIT' file detected")
            break

        if os.path.exists("EXIT-TASK"):
            os.remove("EXIT-TASK")
            info = "exit-task file detected"
            print("\n\t'EXIT-TASK' file detected")
            break

        ##########################################
        # training loop per epoch
        with tqdm(
            enumerate(dataloader_train),
            total=batches_per_epoch,
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            disable=opts["disable"],
        ) as bar:

            ##########################################
            # cycle over mini-batches
            for step, X in bar:

                # necessary to train the model
                model.train(mode=True)

                # predict the value for the input X
                y_pred = model.get_pred(X=X)  # get_pred(model=model,X=X)

                # true value for the value X
                y_real = model.get_real(X=X)

                # compute the loss function
                loss = loss_fn(y_pred, y_real)

                # store the loss function in an array
                train_loss_one_epoch[step] = float(loss)

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                
                # print progress
                bar.set_postfix(
                    epoch=epoch,
                    train=np.mean(train_loss_one_epoch[: step + 1]),
                    val=dataframe.at[epoch - 1, "val"] if epoch != 0 else np.nan,
                )

            ##########################################
            # finished cyclying over mini-batches
            model.eval()

            ##########################################
            # print something to screen
            if hasattr(model,"message"):
                model.message()
            # if model.use_shift:
            #     print("\t!! SHIFT:", model.shift.detach().numpy())
            #     print("\t!! FACTOR:", model.factor.detach().numpy())

            # evaluate model on the test dataset
            # with torch.no_grad():
            # I am not using 'with torch.no_grad()' anymore because
            # maybe it inferferes with the calculation of the forces
            # model.eval()
            # if True:  # with torch.no_grad():

            ##########################################
            # save learning rate
            dataframe.at[epoch,"lr"] = float(optimizer.param_groups[0]["lr"])
            dataframe.at[epoch, "epoch"] = epoch + 1

            ##########################################
            # saving parameters to temporary file
            N = opts["save"]["parameters"]
            if N != -1 and epoch % N == 0:
                savefile = "{:s}/epoch={:d}.pth".format(parameters_folder, epoch)
                print("\tsaving parameters to file '{:s}'".format(savefile))
                torch.save(model.state_dict(), savefile)

            dataframe.at[epoch, "train"] = np.mean(train_loss_one_epoch)
            dataframe.at[epoch, "std"] = np.std(train_loss_one_epoch)

            ##########################################
            # compute the loss function
            # predict the value for the validation dataset
            yval_pred = model.get_pred(all_dataloader_val)
            val_loss = float(loss_fn(yval_pred, yval_real))
            dataframe.at[epoch, "val"] = val_loss            

            ##########################################
            # scheduler
            if scheduler is not None:
                if movingaverage is not None:
                    movingaverage.update(np.log10(val_loss))
                    scheduler.step()
                else :
                    scheduler.step(val_loss)
            
            if not opts["recompute_loss"]:
                dataframe.at[epoch, "ratio"] = (
                    dataframe.at[epoch, "train"] / dataframe.at[epoch, "val"]
                )

            ##########################################
            # set dataframe
            if opts["recompute_loss"]:
                ytrain_pred = model.get_pred(all_dataloader_train) 
                dataframe.at[epoch, "train-2"] = float(loss_fn(ytrain_pred, ytrain_real))
                dataframe.at[epoch, "ratio-2"] = (dataframe.at[epoch, "train-2"] / dataframe.at[epoch, "val"])

            if dataframe.at[epoch, "train"] > opts["thr"]["exit"] and opts["thr"]["exit"] > 0:
                info = "try again"
                break

            dataframe[: epoch + 1].to_csv(savefiles["dataframes"], index=False)

            ##########################################
            # produce learning curve plot
            if epoch >= 1:                
                plot_learning_curves(
                    arrays=dataframe,
                    file=savefiles["images"],
                    title=parameters["name"] if parameters["name"] != "untitled" else None,
                    opts=opts["plot"]["learning-curve"],
                )
            
            ##########################################
            # print progress
            bar.set_postfix(
                epoch=epoch,
                train=dataframe.at[epoch, "train"],
                val=dataframe.at[epoch, "val"],
            )

        ##########################################
        # saving checkpoint to file
        N = opts["save"]["checkpoint"]
        if N != -1 and epoch % N == 0:
            save_checkpoint(checkpoint_file, epoch, model, optimizer, scheduler, movingaverage, movingaverageLR)

        k += 1
    #
    save_checkpoint(checkpoint_file, k - 1, model, optimizer, scheduler, movingaverage, movingaverageLR)

    if info == "all good":

        # removing empty folders
        _folders = copy(folders)
        for k in folders.keys():
            if remove_empty_folder(folders[k], show=False):
                del _folders[k]

        # Important message
        print("\tThe following quantities have been saved to these folders:")
        for k in _folders.keys():
            print("\t{:<20s}: {:<20s}".format(k, folders[k]))

        print("\n\tTraining done!\n")

    # Something wrong happened during the training
    # We will let the user know about that through the variable 'info'
    elif info == "try again":
        print("\n\tTraining stopped: we could try again\n")
        if os.path.exists(checkpoint_file):
            print("\tRemoving checkpoint file '{:s}'".format(checkpoint_file))
            os.remove(checkpoint_file)

    elif info == "exit-task file detected":
        print("\n\t'exit-task' file detected\n")

    return model, dataframe, info
