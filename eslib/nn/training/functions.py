import torch

def save_checkpoint(file,epoch,model,optimizer,scheduler=None,ma=None,maLR=None):
    print("\tsaving checkpoint to file '{:s}'".format(file))
    tosave = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    if scheduler is not None :
        tosave['scheduler_state_dict'] = scheduler.state_dict()
    if ma is not None:
        tosave['ma'] = ma.state_dict()
    if maLR is not None:
        tosave['maLR'] = maLR.state_dict()
    torch.save(tosave, file)

# a useful function
def get_all_dataloader(dataset,make_dataloader):
    """
    Get a data loader for the entire dataset without shuffling.

    This function returns a data loader for the given dataset, allowing access to all data points without shuffling. It's important to set 'shuffle' to False to ensure accurate computation of loss functions.

    Args:
        dataset: Dataset to create the data loader for.

    Returns:
        DataLoader: Data loader for the entire dataset without shuffling.
    """
    # Pay attention!
    # 'shuffle' has to be False!
    # otherwise we will compute the loss function
    # against the wrong data point
    
    return next(iter(make_dataloader(dataset=dataset,
                                    batch_size=len(dataset),
                                    shuffle=False)))

# another useful function
def get_all(model,**argv):
    """
    Get the real values of the entire dataset.

    This function returns the real values of the entire dataset by utilizing the 'get_all_dataloader' function to retrieve a data loader and then extracting the real values from it.

    Args:
        dataset: Dataset to extract real values from.

    Returns:
        torch.Tensor: Real values of the entire dataset.
    """
    all_dataloader = get_all_dataloader(**argv)
    return model.get_real(all_dataloader)