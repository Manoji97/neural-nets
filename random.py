def num2lst(n):
    l = []
    i = 10
    while n >0:
        l.append(n%i)
        n = n//10
    return l[::-1] 

def premutable_next(num): 
    l = num2lst(num)   
    j = len(l)
    c = 1
    for i in range(j-1,0,-1):
        if l[i] <= l[i-1]:
            c +=1
        else:
            break
        
    nl = j - c
    n  = l[nl::][::-1]
    if nl>0:
        for i in range(c):
            if l[nl-1] < n[i]:
                l[nl-1],n[i] = n[i],l[nl-1]
                break
        return l[:nl] + n
    return "its the max"
     