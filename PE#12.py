def divCount(n):
	count = 2
	for k in range(2,n//2+1):
		if n % k == 0:
			count+=1
	return count if n > 1 else 1
i, cont = 0, True
'''while cont:
	i += 1
	if i % 2 == 0:
		tdiv = (divCount(i//2)*divCount(i+1))
	if i % 2 == 1:
		tdiv = (divCount(i)*divCount((i+1)//2))
	if tdiv >= 500:
		print(i)
		cont = False  #Gives Result 12375, i.e the 12375-th triangular number'''

print(sum(i for i in range(1,12376)))
