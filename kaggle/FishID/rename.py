# Simple script to rename fish images so that their filename contains there class label

import os
import glob

fish_id = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
folder = '/foo'                                                    # directory to images

for fish in fish_id:
    k = 0
    for filename in glob.iglob(os.path.join(folder+fish, '*.jpg')):
        os.rename(filename, folder+fish+'/'+fish+str(k)+'.jpg')
        k += 1
