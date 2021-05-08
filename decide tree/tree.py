from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.externals.six import StringIO
import pydotplus

# 初始化决策树模型
dt = tree.DecisionTreeClassifier()

# 导入iris的数据与名字
iris = load_iris()
x = iris.data
y = iris.target
feature_name = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

# 分出0.3的测试集和0.7占比的训练集(random_state随机状态是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。)
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=1)

clf = tree.DecisionTreeClassifier(criterion="entropy",
                                  random_state=0,
                                  # splitter="random",
                                  max_depth=3)
                                  # min_samples_leaf=8)
                                  # max_leaf_nodes=)
clf.fit(X_train, Y_train)
x = clf.feature_importances_
print('[SepalLengthCm '+' SepalWidthCm '+' PetalLengthCm '+' PetalWidthCm]')
print(x)
dot_data = StringIO()

tree.export_graphviz(clf, out_file=dot_data,
                     feature_names=feature_name,
                     class_names=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
                     filled=True,
                     rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

graph.write_png('iris-entropy-deep3.png')
print(clf.score(X_test, Y_test))
