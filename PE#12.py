def divCount(n):
	count = 2
	for k in range(2,n//2+1):
		if n % k == 0:
			count+=1
	return count if n > 1 else 1
t, cont = 0, True
while cont:
	t += 1
	if t % 2 == 0:
		tdiv = (divCount(t//2)*divCount(t+1))
	if t % 2 == 1:
		tdiv = (divCount(t)*divCount((t+1)//2))
	if tdiv >= 500:
		print(t)
		cont = False  #Gives Result 12375, i.e the 12375-th triangular number will be first with over 500 divisors

print(sum(i for i in range(1,t+1)))


#https://projecteuler.net/problem=12
