from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
#iris = load_iris ()
#print(iris.target)
#input()

x=[[0,0,3],[0,1,2],[1,4,2],
   [2,0,0],[3,3,4],[2,1,3],
    [0,2,3],[3,-1,3],[3,1,4],
    [1,0,0],[2,1,1]]
y=[0,0,1,0,1,0,1,0,1,0,0]
gnb = GaussianNB()
gnb.fit(x, y)
predict=gnb.predict([[0,1,3]])
print(predict)