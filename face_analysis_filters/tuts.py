m,n = input("Enter 2 numbers").strip().split(' ')
m,n = [int(m),int(n)]

a = [int(arr_temp) for arr_temp in input().strip().split(' ')]  #enter numbers in a single line and then create array of integer

a = map(int,input().strip().split())	#this will create a map object
a = list(map(int,input().strip().split()))	#creates a list from the map
#another way to take a single line array input of integers with spaces

for i,j,k in matrix3x3:	#i,j,k represent the respective columns in the 3x3 matrix
	print(i,j,k)

