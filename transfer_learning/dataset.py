from lib import *


class Mydataset(Dataset):
    def __init__(self, csv, transform, phase, tem_path='../input/100-bird-species/birds/train/*'):
        self.csv_frame = pd.read_csv(csv)
        self.phase = phase
        self.phase_path = self.csv_frame.loc[self.csv_frame['data set'] == self.phase]
        self.labels = np.array([x[38:] for x in glob.glob(tem_path)])
        self.transform = transform

    def __len__(self):
        return self.phase_path.values.shape[0]

    def __getitem__(self, indx):
        image_path = '../input/100-bird-species/birds/' + self.phase_path.iloc[indx, 1].replace('\\', '/')
        image = Image.open(image_path)
        image_transform = self.transform(image, self.phase)
        label = (self.phase_path.iloc[indx, 2] == self.labels).argmax()

        return image_transform, label


if __name__ == '__main__':
    pass