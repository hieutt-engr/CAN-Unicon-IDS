import torch
import torch.nn.functional as F
import torch.optim as optim

from util import accuracy
from networks.resnet_big import LinearClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

def get_train_features(train_loader, model):
    model.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        # prepare the features and labels
        all_features = []
        all_labels = []
        for idx, (images, labels) in enumerate(train_loader):
            images = images[0]
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]
            features = model.encoder(images)
            all_features.append(features)
            all_labels.append(labels)
    return all_features, all_labels


def get_test_features(test_loader, model):
    model.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        # prepare the features and labels
        all_features = []
        all_labels = []
        for idx, (images, labels) in enumerate(test_loader):
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            features = model.encoder(images)
            all_features.append(features)
            all_labels.append(labels)
    return all_features, all_labels


def set_classifier(features, labels, opt):
    # prepare the classifier
    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_classes).cuda()
    class_opt = optim.SGD(classifier.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    # class_opt = optim.AdamW(classifier.parameters(),
    #                     lr=opt.learning_rate,
    #                     # momentum=opt.momentum,
    #                     weight_decay=opt.weight_decay)
    # train the classifier
    classifier.train()
    for i in range(50):
        for j in range(len(features)):
            feature = features[j]
            y = labels[j]
            y_pred = classifier(feature)
            loss = F.cross_entropy(y_pred, y)
            class_opt.zero_grad()
            loss.backward()
            class_opt.step()
    return classifier


# def test(features, labels, classifier):
#     classifier.eval()
#     y = torch.tensor([]).cuda()
#     y_pred = torch.tensor([]).cuda()
#     with torch.no_grad():
#         for i in range(len(features)):
#             feature = features[i]
#             label = labels[i]
#             pred = classifier(feature)
#             y = torch.cat([y, label], dim=0)
#             y_pred = torch.cat([y_pred, pred], dim=0)
#     acc = accuracy(y_pred, y)
#     print('Top 1 Accuracy:', acc)

def test(features, labels, classifier):
    classifier.eval()
    
    y = torch.tensor([]).cuda()       # True labels
    y_pred = torch.tensor([]).cuda()  # Predicted labels
    
    with torch.no_grad():
        for i in range(len(features)):
            feature = features[i]
            label = labels[i]
            pred = classifier(feature)
            
            y = torch.cat([y, label], dim=0)
            y_pred = torch.cat([y_pred, pred], dim=0)
    
    y_pred_class = torch.argmax(y_pred, dim=1)
    
    acc = accuracy(y_pred, y)
    
    y = y.cpu().numpy()
    y_pred_class = y_pred_class.cpu().numpy()
    conf_matrix = confusion_matrix(y, y_pred_class)

    # acc = accuracy_score(y, y_pred_class)
    f1 = f1_score(y, y_pred_class, average='weighted')
    precision = precision_score(y, y_pred_class, average='weighted', zero_division=0)
    recall = recall_score(y, y_pred_class, average='weighted')

    return acc, f1, precision, recall, conf_matrix

