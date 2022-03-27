#test
import numpy as np
from sklearn import tree
x=np.array([[0,0],[1,0],[0,0],[1,1],[0,1],[1,1],[1,1],[1,0]])
y=np.array([1,0,1,0,1,0,1,1])
clf=tree.DecisionTreeClassifier(criterion='gini')
clf.fit(x,y)


import graphviz
#graphv = graphviz.Source(dot_data)


#print the decision tree in a pdf file
#from sklearn.externals.six import StringIO
from six import StringIO
import pydot

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,
                         filled=True, rounded=True,
                         special_characters=True
                    )
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph[0].write_pdf("test.pdf")