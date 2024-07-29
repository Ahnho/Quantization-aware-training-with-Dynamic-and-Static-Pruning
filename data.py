import torch.utils.data
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet
import torchaudio
import torch.optim as optim
import os
from torch.utils.data import DataLoader, random_split, Dataset
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.transforms import MelSpectrogram




valid_datasets = [
    'cifar10', 'cifar100','imagenet','audio'
]


def _verify_dataset(dataset):
    r"""verify your dataset.  
    If your dataset name is unknown dataset, raise error message..
    """
    if dataset not in valid_datasets:
        msg = "Unknown dataset \'{}\'. ".format(dataset)
        msg += "Valid datasets are {}.".format(", ".join(valid_datasets))
        raise ValueError(msg)
    return dataset


def cifar10_loader(batch_size, num_workers, datapath, image_size=32, cuda=False):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    trainset = CIFAR10(
        root=datapath, train=True, download=True,
        transform=transform_train)
    valset = CIFAR10(
        root=datapath, train=False, download=True,
        transform=transform_val)

    if cuda:
        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=False)
        val_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=False)
    
    return train_loader, val_loader


def cifar100_loader(batch_size, num_workers, datapath, image_size=32, cuda=False):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    trainset = CIFAR100(
        root=datapath, train=True, download=True,
        transform=transform_train)
    valset = CIFAR100(
        root=datapath, train=False, download=True,
        transform=transform_val)

    if cuda:
        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=False)
        val_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=False)
    
    return train_loader, val_loader


def imagenet_loader(batch_size, num_workers, datapath, image_size=224, cuda=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_val = transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    trainset = ImageNet(
        root=datapath, split='train', 
        transform=transform_train)
    
    valset = ImageNet(
        root=datapath, split='val', 
        transform=transform_val)


    if cuda:
        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=False)
        val_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=False)

    return train_loader, val_loader

# class SpeechCommandsDataset(Dataset):
#     def __init__(self, dataset, transform=None):
#         self.dataset = dataset
#         self.transform = transform

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         waveform, sample_rate, label, speaker_id, utterance_number = self.dataset[idx]
#         if self.transform:
#             waveform = self.transform(waveform)
#         return waveform, label

# def pad_sequence(batch):
#     # Make all tensor in a batch the same length by padding with zeros
#     batch = [item.t() for item in batch]
#     batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
#     return batch.permute(0, 2, 1)


# def collate_fn(batch):

#     # A data tuple has the form:
#     # waveform, sample_rate, label, speaker_id, utterance_number

#     tensors, targets = [], []

#     # Gather in lists, and encode labels as indices
#     for waveform, _, label, *_ in batch:
#         tensors += [waveform]
#         targets += [label_to_index(label)]

#     # Group the list of tensors into a batched tensor
#     tensors = pad_sequence(tensors)
#     targets = torch.stack(targets)

#     return tensors, targets
    
    
# def speech_commands_loader(batch_size, num_workers, datapath, image_size=32, cuda=False):
#     # transform = MelSpectrogram()
#     transform = transforms.Compose([
#     torchaudio.transforms.MFCC(sample_rate=16000),  # MFCC 변환
#     transforms.ToTensor()  # 텐서로 변환
# ])
    
#     # Download dataset if not already downloaded
#     dataset = torchaudio.datasets.SPEECHCOMMANDS(datapath, download=True)

#     # Create the custom dataset
#     custom_dataset = SpeechCommandsDataset(dataset, transform=transform)

#     # Split dataset into train and test
#     train_size = int(0.8 * len(custom_dataset))
#     test_size = len(custom_dataset) - train_size
#     train_dataset, val_dataset = random_split(custom_dataset, [train_size, test_size])

#     # Create DataLoaders
#     pin_memory = cuda
    
# #     train_loader = DataLoader(train_dataset,batch_size=256,shuffle=True,collate_fn=collate_fn,num_workers=num_workers,pin_memory=pin_memory)
    
# #     test_loader = DataLoader(val_dataset,batch_size=256,shuffle=False,drop_last=False,collate_fn=collate_fn,num_workers=num_workers,pin_memory=pin_memory)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    
#     return train_loader, val_loader


__classes__ = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "unknown", "silence"]


class SpeechCommandDataset_12(Dataset):
    def __init__(self, datapath, filename, is_training):
        super(SpeechCommandDataset, self).__init__()
        """
        Args:
            datapath: "./datapath"
            filename: train_filename or valid_filename
            is_training: True or False
        """
        self.sampling_rate  = 16000 # 샘플링 레이트
        self.sample_length  = 16000 # 음성의 길이
        self.datapath       = datapath
        self.filename       = filename
        self.is_training    = is_training

        # 카테고리를 키로, 인덱스를 숫자로 매칭
        self.class_encoding = {category: index for index, category in enumerate(__classes__)}
        
        # 음성 augmentations을 위해 데이터 노이즈 백그라운드를 불러옴
        self.noise_path     = os.path.join(self.datapath, "_background_noise_")
        self.noise_dataset = []
        for root, _, filenames in sorted(os.walk(self.noise_path, followlinks = True)):
            for fn in sorted(filenames):
                name, ext = fn.split(".")
                if ext == "wav":
                    self.noise_dataset.append(os.path.join(root, fn)) 
                    # 확장자가 wav인 파일의 경로만 노이즈 데이터셋에 추가
        
        # 음성 데이터의 파일 경로를 라벨과 함께 묶음
        self.speech_dataset = self.combined_path()

                    
    def combined_path(self):
        dataset_list = []
        for path in self.filename:
            category, wave_name = path.split("/") # 음성의 종류와 음성의 이름

            if category in __classes__[:-2]: # "yes부터 go"
                path = os.path.join(self.datapath, category, wave_name)
                dataset_list.append([path, category]) # 음성 경로와 카테고리를 묶어서 페어로 리스트 추가

            elif category == "_silence_": # 음성 경로의 종류가 _silence_ 이면
                dataset_list.append(["silence", "silence"]) # 음성 경로와 카테고리를 묶어서 페어로 리스트 추가

            else: # 만약 음성이 명시적이지도 않고 silence도 아닌 unknown이면 해당 경로는 라벨 unknown과 묶어서 리턴
                path = os.path.join(self.datapath, category, wave_name)
                dataset_list.append([path, "unknown"]) # 음성 경로와 카테고리를 묶어서 페어로 리스트 추가
        return dataset_list

    
    def load_audio(self, speech_path):
        waveform, sr = torchaudio.load(speech_path) # 오디오 로드

        if waveform.shape[1] < self.sample_length: # 오디오의 길이가 sample_length보다 짧으면
            waveform = F.pad(waveform, [0, self.sample_length - waveform.shape[1]]) # 오른쪽으로 16000으로 패딩
        else:
            pass # 아니면 그대로 패스
        
        if self.is_training == True: # 훈련 모드이면 길이 augmentation
            pad_length = int(waveform.shape[1] * 0.1) # 양쪽으로 패딩할 작은 패딩 길이 정의
            waveform   = F.pad(waveform, [pad_length, pad_length]) # 양쪽으로 패딩
            
            # length augmentations을 하고 늘린 길이에서 자를 길이만큼 빼고 + 1 값중 랜덤 offset 설정
            offset   = torch.randint(0, waveform.shape[1] - self.sample_length + 1, size = (1, )).item()
            waveform = waveform.narrow(1, offset, self.sample_length) # 이 중 랜덤으로 sample_length만큼 자름

            if self.noise_augmen == True: # 노이즈 augmentation 옵션
                noise_index = torch.randint(0, len(self.noise_dataset), size = (1,)).item()
                noise, noise_sampling_rate = torchaudio.load(self.noise_dataset[noise_index])

                offset = torch.randint(0, noise.shape[1] - self.sample_length + 1, size = (1, )).item()
                noise  = noise.narrow(1, offset, self.sample_length)
                
                # 노이즈를 sample_length 길이만큼 잘라서 waveform에 붙임
                background_volume = torch.rand(size = (1, )).item() * 0.1
                waveform.add_(noise.mul_(background_volume)).clamp(-1, 1) # -1, 1 이상은 clamp
            else:
                pass # 아니면 패스
        return waveform
    

    def one_hot(self, speech_category): # 카테고리를 숫자 값으로 인코딩하는 함수
        encoding = self.class_encoding[speech_category]
        return encoding
    

    def __len__(self):
        return len(self.speech_dataset)
    

    def __getitem__(self, index):
        self.noise_augmen = self.is_training and random.random() > 0.5 # 노이즈 증강 여부

        speech_path       = self.speech_dataset[index][0] # 패스
        speech_category   = self.speech_dataset[index][1] # 카테고리
        label             = self.one_hot(speech_category)

        if speech_path == "silence": # 경로가 silence이면 torch.zeros로 waveform 생성
            waveform = torch.zeros(1, self.sampling_rate)
        else:
            waveform  = self.load_audio(speech_path) # 해당 소리 파일의 경로로 소리를 불러옴
    
        return waveform, label # (소리, 라벨) 리턴
    


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("/root/data", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]
            
train_set = SubsetSC("training")
test_set = SubsetSC("testing")
waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]
labels = sorted(list(set(datapoint[2] for datapoint in train_set)))   
new_sample_rate = 8000
transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
transformed = transform(waveform)
            
def label_to_index(word):
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))


def index_to_label(index):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]
            
def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def collate_fn(batch):

    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [label_to_index(label)]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets


def speech_commands_loader(batch_size, num_workers, datapath, image_size=256, cuda=False):
    
    train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=cuda
)
    test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=cuda
)
    return train_loader,test_loader, transformed.shape[0] , len(labels)

def DataLoader(batch_size, dataset='cifar10', num_workers=4, datapath='../data', image_size=32, cuda=True):
    """Dataloader for training/validation
    """
    DataSet = _verify_dataset(dataset)
    if DataSet == 'cifar10':
        return cifar10_loader(batch_size, num_workers, datapath, image_size, cuda)
    elif DataSet == 'cifar100':
        return cifar100_loader(batch_size, num_workers, datapath, image_size, cuda)
    elif DataSet == 'imagenet':
        return imagenet_loader(batch_size, num_workers, datapath, image_size, cuda)
    elif DataSet == 'audio':
        return speech_commands_loader(batch_size, num_workers, datapath, image_size, cuda)


if __name__ == '__main__':
    pass