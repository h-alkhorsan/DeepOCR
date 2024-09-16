import torchvision.transforms.functional as F

class ToTensor(object):
    def __call__(self, obj):
        return F.to_tensor(obj)
    
    
class Resize(object):
    def __init__(self, output_size):
        assert(isinstance(output_size, (int, tuple)))
        self.output_size = output_size 

    def __call__(self, obj):
        return F.resize(obj, self.output_size)
    
        # h, w = obj.shape[:2]

        # if isinstance(self.output_size, int):
        #     if h > w:
        #         h_, w_ = self.output_size * h / w, self.output_size
        #     else:
        #         h_, w_ = self.output_size, self.output_size * w / h 
        # else:
        #     h_, w_ = self.output_size

        # obj = skimage.transform.resize(obj, (h_, w_))
        # return obj 
    

class Normalize(object):
    def __init__(self, mean, std):
        assert(isinstance(mean, list))
        assert(isinstance(std, list))
        self.mean = mean 
        self.std = std


    def __call__(self, obj):
        return F.normalize(obj, self.mean, self.std)
        # assert(isinstance(obj), torch.Tensor)
        # mean, std = torch.std_mean(obj)
        # obj_normalized = (obj - mean) / std
        # obj = obj_normalized * self.std + self.mean 
        # return obj 