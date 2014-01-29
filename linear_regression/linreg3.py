from pylab import *

# primero criamos os nossos x e y
# lembre se que o x contem o tamanho dos narizes das pessoas
x = [4.,5.,4.5,3.8]

# o y contem os salarios anuais  
y = [3.,5.,4.7,4.3]

# vamos dar uma olhada nos dados...
scatter(x,y)
xlabel('x')
ylabel('y')
show()

# y = mx + b : funcao da linha. o polyfit vai nos dar o m e o b
# usando toda aquela matematica que nos vimos
(m,b) = polyfit(x,y,1)

# vamos dar uma olhada no m e no b
print b
print m

# isso eh para depois poder testar novos pontos
def predict(val,em,be):
 y = (em*val)+be
 return y 

# vamos testar com o valor do meu nariz x=5
print "valor para x = 5 : ", predict(5,m,b)
 
# vamos dar uma olhada na linha do polifit...
yp = polyval([m,b],x)
plot(x,yp)
scatter(x,y,c='r')
grid(True)
xlabel('x')
ylabel('y')
show() 