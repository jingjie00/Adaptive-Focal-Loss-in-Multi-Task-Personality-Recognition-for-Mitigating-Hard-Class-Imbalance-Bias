def train(loss_function, epochs, trainloader, testloader, validationloader):
    # get the input_size from trainloader
    in_size= trainloader.dataset[0][0].shape[0]

    network = CustomNetwork(input_size=in_size)
    network = best_device(network)

    model = CustomModel(network)
        
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

    for e in range(epochs):
        # a dictionary that store the training loss, validation loss, train_size, validation_size, TP, FP, TN, FN
        running_info = {'train_loss':0, 'validation_loss':0, 'train_size':0, 'validation_size':0, 'TP':0, 'FP':0, 'TN':0, 'FN':0}

        # set to training mode
        network.train(True)

        # per epoch training activity
        for inputs, labels in trainloader:

            # clear all the gradient to 0
            optimizer.zero_grad()

            inputs,labels = best_device(inputs, labels)

            # forward propagation
            outs = network(inputs)
            outs = outs.view(-1)
            
            # compute loss
            loss = loss_function.forward(inputs=outs, targets=labels)
            
            # backpropagation
            loss.backward()

            # update w
            optimizer.step()

            # update running_info
            running_info['train_loss'] += loss.item()*labels.size(0)
            running_info['train_size'] += labels.size(0)


        # Turn off training mode for reporting validation loss
        network.train(False)

        # per epoch validation activity
        for inputs, labels in validationloader:

 
            inputs,labels = best_device(inputs, labels)

            # forward propagation
            outs = network(inputs)
            outs = outs.view(-1)

            # update running_info
            running_info['validation_loss'] += loss.item()*labels.size(0)
            running_info['validation_size'] += labels.size(0)

            preds = (outs > 0.5).type(torch.FloatTensor)
            running_info['TP'],running_info['FP'],running_info['TN'],running_info['FN'] = e_confusion_matrix(preds,labels)

        
        train_loss = running_info['train_loss']/running_info['train_size']
        validation_loss = running_info['validation_loss']/running_info['validation_size']

        confusion_matrix = running_info['TP'],running_info['FP'],running_info['TN'],running_info['FN']
        regular_accuracy,balanced_accuracy = e_accuracy(confusion_matrix)
        

        print(f'[Epoch {e + 1:2d}/{epochs:d}]: train_loss = {train_loss:.4f}, validation_loss = {validation_loss:.4f}, RA = {regular_accuracy:.4f}, BA: {balanced_accuracy:.4f}, CM:{confusion_matrix}')

        model.update(network, epochs = e+1, ba = balanced_accuracy, ra=regular_accuracy)

     # per epoch test activity
    for inputs, labels in testloader:


        inputs,labels = best_device(inputs, labels)

        # forward propagation
        outs = network(inputs)
        outs = outs.view(-1)

        # update running_info
        running_info['test_loss'] += loss.item()*labels.size(0)
        running_info['test_size'] += labels.size(0)

        preds = (outs > 0.5).type(torch.FloatTensor)
        running_info['TP'],running_info['FP'],running_info['TN'],running_info['FN'] = e_confusion_matrix(preds,labels)
    
    
    return model