#!/usr/bin/python3

'''This is my solution to Project Euler Problem 13.
   https://projecteuler.net/problem=13
   I have saved the list of 50-digit numbers as a .txt file'''

num_list = open('/home/james/py/euler1.txt', 'r')

def ten_split(n):
    '''Takes a large integer (>10 digits) as a string and returns
       as a pair the first 25 digits and remaining 25 as
       a decimal'''
    return [int(n[:25]), float('0.'+n[25:])]


split_nums = [ten_split(line.strip('\n')) for line in num_list]

x, y = sum(e[0] for e in split_nums), int(sum(e[1] for e in split_nums))

print('The answer is: ', str(x+y)[:10])
