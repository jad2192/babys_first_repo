# Data Augmentation Function

def aug_images(fp = 'filepath', spec, iter):
  '''Data augmentation function.
     fp: filepath to the images to augment. Assuming each species in its own folder and for each
         species folder (e.g fp/ALB) also assumes there is a dual folder named fp/augemented in
         which to place the augemented images.
         
     spec: ALB, BET, LAG ... etc. The name of species to augment.
     
     iter: Number of times to loop, more means more augmentation. Adjust to compensate
           for class imbalances.'''
           
  fish_lists = os.listdir(fp+'/'+spec) 

  for fish in fish_lists:
      lab = '/'+fish[:3]+'/'
      cur_fish = PIL.Image.open(fp + lab + fish)
      cur_fish = cur_fish.resize((700,700),PIL.Image.ANTIALIAS)
    
      for k in range(2):
        
          left = np.random.choice(list(range(24,75)))
          upper = np.random.choice(list(range(24,75)))
          right = 699 - (100 - left)
          lower = 699 - (100 - upper)
          cur_t = cur_fish.crop((left,upper,right,lower))
    
          cur_t.save(fp=fp+'/augmented/'+fish[:-4]+'t'+str(k)+'.jpg')
          cur_sh = cur_t.filter(PIL.ImageFilter.SHARPEN)
          cur_sh.save(fp=fp+'/augmented/'+fish[:-4]+'sh'+str(k)+'.jpg')
          cur_e = cur_t.filter(PIL.ImageFilter.EDGE_ENHANCE)
          cur_e.save(fp=fp+'/augmented/'+fish[:-4]+'e'+str(k)+'.jpg')
      cur_fish.close()
