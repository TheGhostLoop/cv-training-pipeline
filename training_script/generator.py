def generate_training_script(config):
    return f"""
# Auto-generated training script

DATASET_PATH = "{config['dataset_path']}"
MODEL_NAME = "{config['model_name']}"
DATASET_STRUCTURE = "{config['dataset_structure']}"
TRAINING_MODE = "{config['training_mode']}"
BATCH_SIZE = {config['batch_size']}
EPOCHS = {config['epochs']}


from datetime import datetime
import os


BASE_DIR = "/content/datasets"
os.makedirs(BASE_DIR, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
DATASET_DIR = os.path.join(BASE_DIR, f"dataset_{{TIMESTAMP}}")


os.makedirs(DATASET_DIR, exist_ok=True)

import zipfile,requests,io
choice = input("Enter source 'file' for 'url' \\nYour choice:: ").lower().strip()

try:
    if (choice=='file'):
        name = input("Enter file name including (.zip): ")
        with zipfile.ZipFile(name,"r") as zip_ref:
            zip_ref.extractall(DATASET_DIR)
            print("File extracted successfully to the datasets directory") 


    elif (choice=='url'):
        url = input("Enter url:: ")
        print("Downloading and extracting zip file please wait!")
        r = requests.get(url)
        zip_bytes = io.BytesIO(r.content)
        with zipfile.ZipFile(zip_bytes,"r") as zip_ref:
            zip_ref.extractall(DATASET_DIR)
            print("File extracted successfully to the datasets directory") 

    else:
        print("Wrong choice! re run the script.")

except Exception as e:
    print(f"Error occured: {{e}}")



def find_actual_dataset_root(base_dir, structure_type):

    # First check base_dir itself
    contents = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    # Case 1: pre-split dataset
    if structure_type == "pre_split":
        if "train" in contents:
            return base_dir

        # Look one level deeper
        for d in contents:
            subdir = os.path.join(base_dir, d)
            sub_contents = os.listdir(subdir)
            if "train" in sub_contents:
                return subdir

    # Case 2: single-folder dataset
    elif structure_type == "single_folder":
        # If base_dir directly contains class folders
        if len(contents) >= 2:
            return base_dir

        # Look one level deeper
        for d in contents:
            subdir = os.path.join(base_dir, d)
            sub_contents = [
                x for x in os.listdir(subdir)
                if os.path.isdir(os.path.join(subdir, x))
            ]
            if len(sub_contents) >= 2:
                return subdir

    raise ValueError("Could not locate a valid dataset root.")

DATA_ROOT = find_actual_dataset_root(
                DATASET_DIR,
                DATASET_STRUCTURE
)
print(f"Data root is {{DATA_ROOT}}")


from torchvision import transforms,models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,random_split
import torch
import json

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
# ---------------- Transforming images --------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_ALIASES = ['train', 'training']
VAL_ALIASES = ['val', 'valid', 'validation']
TEST_ALIASES = ['test', 'testing']

def detect_splits(dataset_root):
    subdirs = {{
        d.lower(): os.path.join(dataset_root, d)
        for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
    }}

    train_dir = None
    val_dir = None
    test_dir = None

    for name, path in subdirs.items():
        if any(alias == name for alias in TRAIN_ALIASES):
            train_dir = path
        elif any(alias == name for alias in VAL_ALIASES):
            val_dir = path
        elif any(alias == name for alias in TEST_ALIASES):
            test_dir = path

    return train_dir, val_dir, test_dir

if(DATASET_STRUCTURE == "pre_split"):
    train_dir, val_dir, test_dir = detect_splits(DATA_ROOT)

    if train_dir is None:
        raise ValueError("No training folder found in pre-split dataset.")

    # If val is missing but test exists → use test as validation
    if val_dir is None and test_dir is not None:
        val_dir = test_dir

    # If both val and test missing → error
    if val_dir is None:
        raise ValueError("No validation/test folder found.")


transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = [0.485, 0.456, 0.406],
                std  = [0.229, 0.224, 0.225]
            )
])

if(DATASET_STRUCTURE=="pre_split"):
    train_dataset = ImageFolder(
        root=train_dir,
        transform=transform
    )
    test_dataset = ImageFolder(
        root=val_dir,
        transform=transform
    )
    num_classes = len(train_dataset.classes)
    classes = train_dataset.classes

    with open("class_mapping.json", "w") as f:
        json.dump(train_dataset.class_to_idx, f)



else: #for single folder -> split using random_split
    dataset = ImageFolder(
        root=DATA_ROOT,
        transform=transform
    )
    num_classes = len(dataset.classes)
    classes = dataset.classes

    with open("class_mapping.json", "w") as f:
        json.dump(dataset.class_to_idx, f)


    train_size = int(0.8*len(dataset))
    test_size = int(len(dataset)-train_size)
    train_dataset,test_dataset = random_split(dataset,[train_size,test_size])


train_loader = DataLoader(dataset=train_dataset,shuffle=True,batch_size=BATCH_SIZE,num_workers=2,pin_memory=True)
test_loader = DataLoader(dataset=test_dataset,shuffle=False,batch_size=BATCH_SIZE,num_workers=2,pin_memory=True)


print("Starting training model on: ",device)
# ------------ MODEL SELECTION ----------------

if(MODEL_NAME=="resnet18"):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features,num_classes)

else:
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features,num_classes)

# ------------ FREEZING AND FINE TUNING ------------

for param in model.parameters():
    param.requires_grad = False


if(TRAINING_MODE=="Final Head Only" and MODEL_NAME=="resnet18"):
    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer = optim.Adam([
        {{'params':model.fc.parameters(),'lr':1e-3}}
    ])

elif(TRAINING_MODE=="Layer4 Unfreeze + Final Head" and MODEL_NAME=="resnet18"):
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer = optim.Adam([
        {{'params':model.layer4.parameters(),'lr':1e-4}},
        {{'params':model.fc.parameters(),'lr':1e-3}}
    ])
    

elif(TRAINING_MODE=="Final Head Only" and MODEL_NAME=="mobilenetv2"):
    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer = optim.Adam([
        {{'params':model.classifier.parameters(),'lr':1e-3}}
    ])

else:
    for param in model.features[-6:].parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer = optim.Adam([
        {{'params':model.features[-6:].parameters(),'lr':1e-4}},
        {{'params':model.classifier.parameters(),'lr':1e-3}}
    ])

total_param,train_param = 0,0
for name,param in model.named_parameters():
    total_param+=param.numel()
    if param.requires_grad:
        train_param+=param.numel()
        print(f"Trainable: {{name}}")

print(f"Total parameters:: {{total_param}}")
print(f"Trainable parameters:: {{train_param}}")


criterion = nn.CrossEntropyLoss()
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    patience=2,
    factor= 0.1
)
model.to(device)
epochs = EPOCHS
TARGET_ACC = 99.9
best_val_acc = 0.0
train_acc_list,val_acc_list = [],[]

for epoch in range(epochs):
    start = time.perf_counter()
    model.train()
    training_loss,training_total,training_correct=0,0,0

    for images,labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_total+=labels.size(0)
        training_loss+=loss.item()
        _,prediction = torch.max(logits,1)
        training_correct+=(prediction==labels).sum().item()
        training_time = time.perf_counter() - start

    training_accuracy = 100*training_correct / training_total
    training_loss /= len(train_loader)

    model.eval()
    val_loss,val_total,val_correct=0,0,0
    with torch.no_grad():
        for images,labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits,labels)

            val_total+=labels.size(0)
            val_loss+=loss.item()
            _,prediction = torch.max(logits,1)
            val_correct+= (prediction==labels).sum().item()


        val_accuracy = 100*val_correct / val_total
        val_loss /= len(test_loader)


        train_acc_list.append(training_accuracy)
        val_acc_list.append(val_accuracy)

    if(epoch%1==0):
        print("---------------------")
        for i,group in enumerate(optimizer.param_groups):
            print(f"Group {{i}}, LR: {{group['lr']}}")
        print(f"Training time this epochs:: {{training_time}}")
        print(f"Epoch: {{epoch+1}}/{{epochs}}:-")
        print(f"Traning loss: {{training_loss}} ")
        print(f"Traning Accuracy: {{training_accuracy}} ")
        print(f"Val loss: {{val_loss}} ")
        print(f"Val Accuracy: {{val_accuracy}} ")

    scheduler.step(val_loss)

    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        torch.save(model, "best_model.pth")

    if(val_accuracy>=TARGET_ACC):
        print(f"Val accuracy reached target!.")
        break

epochs_range = range(1,len(train_acc_list)+1)
import matplotlib.pyplot as plt


print(f"Best trained model saved at ::best_model.pth\\nUse inference script to make predictions now.")
plt.figure(figsize=(8,5))
plt.plot(epochs_range,train_acc_list,label="Training Accuracy",color="blue",marker="o")
plt.plot(epochs_range,val_acc_list,label="Validation Accuracy",color="green",marker="o")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training Vs Validation Accuracy")
plt.legend()
plt.savefig("Training_curves.png")
plt.close()
print("Accuracy graph saved at training_curves.png")

"""