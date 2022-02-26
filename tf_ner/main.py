from src.utils.config import config
from src.recongizer import Recongnizer


if __name__ == '__main__':
    recongnizer = Recongnizer(config)
    recongnizer.fit()
    # x = ['朝阳区小关北里000-0号', '朝阳区惠新东街00号', '朝阳区多处营业网点']
    # recongnizer.predict(x)
