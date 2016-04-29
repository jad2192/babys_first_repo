#!/usr/bin/env python3

#https://projecteuler.net/problem=8

def prodCheck(s,k):
  c, p = s[k:k+13], 1
  for x in c:
    p *= int(x)
  return p

num_file = open('/home/james/py/euler1.txt', 'r')

num_string = ''

for line in num_file:
  num_string += line.strip('\n')

res = 1

for k in range(len(num_string)-13):
  res = max(res,prodCheck(num_string,k))

print("The answer is: ", res)
