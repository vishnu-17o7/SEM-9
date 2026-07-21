chars = ["a","a","b","b","c","c","c","a","a","a","d"]
count =1
f =""
final_count=0
for i in range(len(chars)-1):
    if chars[i] == chars[i+1]:
       count +=1
    else:
        if count ==1:
            final+= chars[i]
        else:
            final = chars[i]+ str(count)
        count =1
        f = f+final
    
f = f + chars[-1] + str(count)
print(f)
for i in f:
    final_count +=1

print(final_count)

