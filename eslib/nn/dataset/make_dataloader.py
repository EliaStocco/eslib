from torch_geometric.loader import DataLoader

def make_dataloader(dataset,batch_size=1,shuffle=True,drop_last=True):

        if batch_size == -1 :
                batch_size = len(dataset)

        if batch_size > len(dataset):
                raise ValueError("'batch_size' is greater than the dataset size")
            
        dataloader = DataLoader(dataset,\
                                batch_size=batch_size,\
                                drop_last=drop_last,\
                                shuffle=shuffle)
    
        return dataloader