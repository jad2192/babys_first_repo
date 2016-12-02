#!/usr/bin/python3
'''
Command line battleship game I programmed as a self-learning exercise while first learning to program in python.
'''
#init
import random
#generate coordinates
cord = []
for i in range(1, 101):
  if i % 10 == 0:
    col = 10
    row = i // 10
  else:
    col = i % 10
    row = (i // 10) + 1
  cord = cord + [[row, col]]

myremain = list(cord)
compremain = list(cord)
myhits, comphits, myships, compships, myguess, compguess = [], [], [], [], [], []
side = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
#Define the grid printing functions
def printmygrid():
  
  for j in range(11):
    if j != 10:  
      if j == 0:
        print('   ', end='', sep='')    
      else:
        print('{:<3}'.format(str(j)), end='', sep='')
    else:
      print('{:<3}'.format('10'), sep='')
  for i in range(100):
    if cord[i][1] != 10:
      if cord[i][1] == 1:
        if (cord[i] in compguess) and (cord[i] not in myhits):
          print('{:<2}'.format(side[i // 10]), end='', sep='')
        elif (cord[i] in myhits):
          print('{:<2}'.format(side[i // 10]) + '|X|', end='', sep='')
        elif (cord[i] in myships) and (cord[i] not in myhits):
          print('{:<2}'.format(side[i // 10]) + '|O|', end='', sep='')
        else:
          print('{:<2}'.format(side[i // 10]) + '|_|', end='', sep='')
        
      if (cord[i] not in myhits) and (cord[i] not in myships) and (cord[i][1] != 1) and (cord[i] not in compguess):
        print('|_|', end='', sep='')
      elif (cord[i] in myships) and (cord[i] not in myhits) and (cord[i][1] != 1):
        print('|O|', end='', sep='')
      elif (cord[i] in myhits) and (cord[i][1] != 1):
        print('|X|', end='', sep='')
      elif (cord[i] in compguess) and (cord[i] not in myhits) and (cord[i] != 1):
        print('|m|', end='', sep='')
    if cord[i][1] == 10:
      if (cord[i] not in myhits) and (cord[i] not in myships) and (cord[i] not in compguess):
        print('|_|', sep='')
      elif (cord[i] in myships) and (cord[i] not in myhits) and (cord[i] not in compguess):
        print('|O|', sep='')
      elif (cord[i] in myhits):
        print('|X|', sep='')
      elif (cord[i] in compguess):
        print('|m|', sep='')

def printcompgrid():
  for j in range(11):
      if j != 10:  
        if j == 0:
          print('   ', end='', sep='')    
        else:
          print('{:<3}'.format(str(j)), end='', sep='')
      else:
        print('{:<3}'.format('10'), sep='')
  for i in range(100):
    if cord[i][1] != 10:
      if cord[i][1] == 1:
        if (cord[i] not in comphits) and (cord[i] not in myguess):
          print('{:<2}'.format(side[i // 10]) + '|_|', end='', sep='')
        elif (cord[i] in comphits):
          print('{:<2}'.format(side[i // 10]) + '|X|', end='', sep='')
        elif (cord[i] in myguess) and (cord[i] not in comphits):
          print('{:<2}'.format(side[i // 10]) + '|m|', end='', sep='')
      if (cord[i] not in comphits) and (cord[i][1] != 1) and (cord[i] not in myguess):
        print('|_|', end='', sep='')
      elif (cord[i] in comphits) and (cord[i][1] != 1):
        print('|X|', end='', sep='')
      elif (cord[i] in myguess) and (cord[i][1] != 1):
        print('|m|', end='', sep='')
    if cord[i][1] == 10:
      if (cord[i] not in comphits) and (cord[i] not in myguess) :
        print('|_|', sep='')
      elif (cord[i] in myguess) and (cord[i] not in comphits):
        print('|m|', sep='')
      elif cord[i] in comphits:
        print('|X|', sep='')
      else:
        print('|_|', sep='')


#Define the user input conversion

def conv(nums):
  nums[0] = (ord(nums[0]) - ord('A')) + 1
  return [nums[0], nums[1]]

def cordin():
  r = 0
  c = 0
  while r not in side:
    r = str(input("Enter the row letter of your choice: "))
  while c not in range(1,11):
    c = int(input("Enter the column number of your choice: "))
  return conv([r, c])

#Get the player's ships functions

def getPT():
  
  print("""Place your PT boat (2 spaces) by choosing the orientaion (horizontal or vertical) and starting point.
  The boat will fill by default to the right (horizontal) or below (vertical) the starting point
  unless there is not enough space in which case it will fill in the other direction.""")
  orient = 0
  while orient not in ['V', 'H']:
    orient = str(input("Place vertical or horizontal? [V/H] "))
  start = []
  print("Choose the starting coordinate.")
  while (start not in cord):
    start = cordin()
  if orient == 'V':
    if start[0] < 10:
      PT = [start, [start[0]+1,start[1]]]
    elif start[0] == 10:
      PT = [[9, start[1]], start]
  elif orient == 'H':
    if start[1] < 10:
      PT = [start, [start[0], start[1]+1]]
    elif start[1] == 10:
      PT = [start, [start[0], 9]]
  return PT

def getDest():
  print("""Place your destroyer (3 spaces) by choosing the orientaion (horizontal or vertical) and starting point.
  The boat will fill by default to the right (horizontal) or below (vertical) the starting point
  unless there is not enough space in which case it will fill in the other direction.""")
  orient = 0
  while orient not in ['V', 'H']:
    orient = str(input("Place vertical or horizontal? [V/H] "))
 
  valid = False
  while not valid:
    dest = []
    start = []
    print("Choose the starting coordinate.")
    while (start not in cord):
      start = cordin()
    if orient == 'V':
      if start[0] < 9:
        dest = [start, [start[0]+1, start[1]], [start[0]+2, start[1]]]
      else:
        dest = [start, [start[0]-1, start[1]], [start[0]-2, start[1]]]
    if orient == 'H':
      if start[1] < 9:
        dest = [start, [start[0], start[1]+1], [start[0], start[1]+2]]
      else:
        dest = [start, [start[0], start[1]-1], [start[0], start[1]-2]]
    count = 0
    for x in dest:
      if x in myships:
        count = count + 1
    valid = (count == 0)   
      
  return dest

def getSub():
  print("""Place your submarine (3 spaces) by choosing the orientaion (horizontal or vertical) and starting point.
  The boat will fill by default to the right (horizontal) or below (vertical) the starting point
  unless there is not enough space in which case it will fill in the other direction.""")
  orient = 0
  while orient not in ['V', 'H']:
    orient = str(input("Place vertical or horizontal? [V/H] "))
 
  valid = False
  while not valid:
    sub = []
    start = []
    print("Choose the starting coordinate.")
    while (start not in cord):
      start = cordin()
    if orient == 'V':
      if start[0] < 9:
        sub = [start, [start[0]+1, start[1]], [start[0]+2, start[1]]]
      else:
        sub = [start, [start[0]-1, start[1]], [start[0]-2, start[1]]]
    if orient == 'H':
      if start[1] < 9:
        sub = [start, [start[0], start[1]+1], [start[0], start[1]+2]]
      else:
        sub = [start, [start[0], start[1]-1], [start[0], start[1]-2]]
    count = 0
    for x in sub:
      if x in myships:
        count = count + 1
    valid = (count == 0)   
      
  return sub

def getBat():
  print("""Place your Battleship (4 spaces) by choosing the orientaion (horizontal or vertical) and starting point.
  The boat will fill by default to the right (horizontal) or below (vertical) the starting point
  unless there is not enough space in which case it will fill in the other direction.""")
  orient = 0
  while orient not in ['V', 'H']:
    orient = str(input("Place vertical or horizontal? [V/H] "))
 
  valid = False
  while not valid:
    bat = []
    start = []
    print("Choose the starting coordinate.")
    while (start not in cord):
      start = cordin()
    if orient == 'V':
      if start[0] < 8:
        bat = [start, [start[0]+1, start[1]], [start[0]+2, start[1]], [start[0]+3, start[1]]]
      else:
        bat = [start, [start[0]-1, start[1]], [start[0]-2, start[1]], [start[0]-3, start[1]]]
    if orient == 'H':
      if start[1] < 8:
        bat = [start, [start[0], start[1]+1], [start[0], start[1]+2], [start[0], start[1]+3]]
      else:
        bat = [start, [start[0], start[1]-1], [start[0], start[1]-2], [start[0], start[1]-3]]
    count = 0
    for x in bat:
      if x in myships:
        count = count + 1
    valid = (count == 0)   
      
  return bat

def getAir():
  print("""Place your aircraft carrier (5 spaces) by choosing the orientaion (horizontal or vertical) and starting point.
  The boat will fill by default to the right (horizontal) or below (vertical) the starting point
  unless there is not enough space in which case it will fill in the other direction.""")
  orient = 0
  while orient not in ['V', 'H']:
    orient = str(input("Place vertical or horizontal? [V/H] "))
 
  valid = False
  while not valid:
    air = []
    start = []
    print("Choose the starting coordinate.")
    while (start not in cord):
      start = cordin()
    if orient == 'V':
      if start[0] < 7:
        air = [start, [start[0]+1, start[1]], [start[0]+2, start[1]], [start[0]+3, start[1]], [start[0]+4, start[1]]]
      else:
        air = [start, [start[0]-1, start[1]], [start[0]-2, start[1]], [start[0]-3, start[1]], [start[0]-4, start[1]]]
    if orient == 'H':
      if start[1] < 7:
        air = [start, [start[0], start[1]+1], [start[0], start[1]+2], [start[0], start[1]+3], [start[0], start[1]+4]]
      else:
        air = [start, [start[0], start[1]-1], [start[0], start[1]-2], [start[0], start[1]-3], [start[0], start[1]-4]]
    count = 0
    for x in air:
      if x in myships:
        count = count + 1
    valid = (count == 0)   
      
  return air

#Generate computer's boats

def compPT():
  vh = ['V', 'H']
  orient = random.choice(vh)
  start = random.choice(cord)
  if orient == 'V':
    if start[0] < 10:
      PT = [start, [start[0]+1,start[1]]]
    elif start[0] == 10:
      PT = [[9, start[1]], start]
  elif orient == 'H':
    if start[1] < 10:
      PT = [start, [start[0], start[1]+1]]
    elif start[1] == 10:
      PT = [start, [start[0], 9]]
  return PT

def compDest():
  vh = ['V', 'H']
  orient = random.choice(vh)
  valid = False
  while not valid:
    dest = []
    start = random.choice(cord)
    print("Choose the starting coordinate.")
    if orient == 'V':
      if start[0] < 9:
        dest = [start, [start[0]+1, start[1]], [start[0]+2, start[1]]]
      else:
        dest = [start, [start[0]-1, start[1]], [start[0]-2, start[1]]]
    if orient == 'H':
      if start[1] < 9:
        dest = [start, [start[0], start[1]+1], [start[0], start[1]+2]]
      else:
        dest = [start, [start[0], start[1]-1], [start[0], start[1]-2]]
    count = 0
    for x in dest:
      if x in compships:
        count = count + 1
    valid = (count == 0)   
      
  return dest

def compSub():
  vh = ['V', 'H']
  orient = random.choice(vh)
  valid = False
  while not valid:
    sub = []
    start = random.choice(cord)
    if orient == 'V':
      if start[0] < 9:
        sub = [start, [start[0]+1, start[1]], [start[0]+2, start[1]]]
      else:
        sub = [start, [start[0]-1, start[1]], [start[0]-2, start[1]]]
    if orient == 'H':
      if start[1] < 9:
        sub = [start, [start[0], start[1]+1], [start[0], start[1]+2]]
      else:
        sub = [start, [start[0], start[1]-1], [start[0], start[1]-2]]
    count = 0
    for x in sub:
      if x in compships:
        count = count + 1
    valid = (count == 0)   
      
  return sub

def compBat():
  vh = ['V', 'H']
  orient = random.choice(vh)
  valid = False
  while not valid:
    bat = []
    start = random.choice(cord)
    if orient == 'V':
      if start[0] < 8:
        bat = [start, [start[0]+1, start[1]], [start[0]+2, start[1]], [start[0]+3, start[1]]]
      else:
        bat = [start, [start[0]-1, start[1]], [start[0]-2, start[1]], [start[0]-3, start[1]]]
    if orient == 'H':
      if start[1] < 8:
        bat = [start, [start[0], start[1]+1], [start[0], start[1]+2], [start[0], start[1]+3]]
      else:
        bat = [start, [start[0], start[1]-1], [start[0], start[1]-2], [start[0], start[1]-3]]
    count = 0
    for x in bat:
      if x in compships:
        count = count + 1
    valid = (count == 0)   
      
  return bat

def compAir():
  vh = ['V', 'H']
  orient = random.choice(vh)
  valid = False
  while not valid:
    air = []
    start = random.choice(cord)
    if orient == 'V':
      if start[0] < 7:
        air = [start, [start[0]+1, start[1]], [start[0]+2, start[1]], [start[0]+3, start[1]], [start[0]+4, start[1]]]
      else:
        air = [start, [start[0]-1, start[1]], [start[0]-2, start[1]], [start[0]-3, start[1]], [start[0]-4, start[1]]]
    if orient == 'H':
      if start[1] < 7:
        air = [start, [start[0], start[1]+1], [start[0], start[1]+2], [start[0], start[1]+3], [start[0], start[1]+4]]
      else:
        air = [start, [start[0], start[1]-1], [start[0], start[1]-2], [start[0], start[1]-3], [start[0], start[1]-4]]
    count = 0
    for x in air:
      if x in compships:
        count = count + 1
    valid = (count == 0)   
      
  return air


# Begin the game!
print("\nWelcome to Battleship! Prepare for War!!!")
print("Begin by placing your ships.")
print("\nHere is your board.")

# Place the ships
printmygrid()
PT = getPT()
myships = myships + PT
print("\nHere is your board so far.")
printmygrid()
dest = getDest()
myships = myships + dest
print("\nHere is your board so far.")
printmygrid()
sub = getSub()
myships = myships + sub
print("\nHere is your board so far.")
printmygrid()
bat = getBat()
myships = myships + bat
print("\nHere is your board so far.")
printmygrid()
air = getAir()
myships = myships + air
print("\nHere is your complete board.")
printmygrid()

print("The computer will now place their ships.")

cPT = compPT()
compships = compships + cPT
cdest = compDest()
compships = compships + cdest
csub = compSub()
compships = compships + csub
cbat = compBat()
compships = compships + cbat
cair = compAir()
compships = compships + cair

# Begin the turns
while (len(myships) != len(myhits)) and (len(comphits) != len(compships)):
  print("\nYour attack board.")
  printcompgrid()
  print("\nYour board.")
  printmygrid()
  print("\nEnter your attack coordinates. ")
  myattack = cordin()
  while (myattack not in myremain):
    myattack = cordin()
  myremain.remove(myattack)
  myguess = myguess + [myattack]
  if myattack in compships:
    print("Hit!")
    comphits = comphits + [myattack]
  else:
    print("Miss")
  print("The enemy is now attacking!")
  compattack = random.choice(compremain)
  compremain.remove(compattack)
  compguess = compguess + [compattack]
  if compattack in myships:
    if compattack in PT:
      print("Our PT boat is hit!")
    elif compattack in dest:
      print("Our destroyer is hit!")
    elif compattack in sub:
      print("Our submarine is hit!")
    elif compattack in bat:
      print("Our battleship is hit!")
    else:
      print("Our aircraft carrier is hit!")
    myhits = myhits + [compattack]
  else:
    print("They've missed!")
  

  
  
if (len(myhits) == len(myships)) and (len(comphits) != len(compships)):
  print("You Lose...")
else:
  print("You Won! Good Job!")

