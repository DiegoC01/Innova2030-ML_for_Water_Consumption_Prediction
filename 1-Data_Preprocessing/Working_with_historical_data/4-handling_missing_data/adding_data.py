f= open("july_missing_data.csv","w+")

inicio_1 = 1688832264
fin_1 = 1689100420

inicio_2 = 1689622713
fin_2 = 1689719805

while(inicio_1 <= fin_1):
     f.write(str(inicio_1)+",Innova-1,0.0\n")
     inicio_1 += 1

while(inicio_2 <= fin_2):
     f.write(str(inicio_2)+",Innova-1,0.0\n")
     inicio_2 += 1


