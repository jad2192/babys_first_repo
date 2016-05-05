#!/usr/bin/python3

'''This is my solution to Project Euler Problem 14.
   https://projecteuler.net/problem=14'''

def collatz_length(n):
    '''Outputs the length of collatz sequence'''
    l, s = 1, n
    while s > 1:
        s = (s//2)*(s%2==0) + (3*s+1)*(s%2==1)
        l += 1
    return l

ans_length, ans = 1, 1

for k in range(1,1000000):
    if collatz_length(k) > ans_length:
        ans_length, ans = collatz_length(k), k

print('The answer is: ', ans)
