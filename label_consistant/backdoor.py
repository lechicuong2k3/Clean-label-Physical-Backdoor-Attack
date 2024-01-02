import numpy as np
from PIL import Image
import random

class CLBD(object):
    """ Label-Consistent Backdoor Attacks.

    Reference:
    [1] "Label-consistent backdoor attacks."
    Turner, Alexander, et al. arXiv 2019.

    Args:
        trigger_path (str): Trigger path.
    """

    def __init__(self, trigger_path):
        with open(trigger_path, "rb") as f:
            trigger_ptn = Image.open(f).convert("RGB")
        self.trigger_ptn = np.array(trigger_ptn)
        self.trigger_loc = np.nonzero(self.trigger_ptn)

    def __call__(self, img):
        return self.add_trigger(img)

    def add_trigger(self, img):
        """Add `trigger_ptn` to `img`.

        Args:
            img (numpy.ndarray): Input image (HWC).
        
        Returns:
            poison_img (np.ndarray): Poison image (HWC).
        """
        
        dark_image = np.zeros((224, 224, 3), dtype=np.uint8)
        trigger_x_size = self.trigger_ptn.shape[0]
        trigger_y_size = self.trigger_ptn.shape[1]

        # Choose the center coordinates for the trigger on the dark image
        center_x_dark = img.shape[0] // 2
        center_y_dark = img.shape[1] // 2
        x = random.randrange(center_x_dark+25, center_x_dark+45)
        y = random.randrange(center_y_dark-10, center_y_dark+10)
        
        start_x_dark = x - trigger_x_size // 2
        end_x_dark = x + trigger_x_size // 2 +1
        start_y_dark = y - trigger_y_size // 2
        end_y_dark = y + trigger_y_size // 2 
        
        # Create the square trigger on the dark image
        dark_image[start_x_dark:end_x_dark, start_y_dark:end_y_dark, :] = self.trigger_ptn
        
        trigger_loc = np.nonzero(dark_image)
        img[trigger_loc] = 0
        poison_img = img + dark_image

        return poison_img