
# pip install graphviz

from graphviz import Digraph

g = Digraph('splits', filename='data_splits', format='svg')
g.attr(rankdir='LR', nodesep='0.4', ranksep='0.6')

def method(col, title, bf_flag):
    with g.subgraph(name=f'cluster_{col}') as c:
        c.attr(label=title, style='rounded')
        c.node(f'data{col}', 'Data\n(c0–c3)', shape='box',
               style='filled', fillcolor='#dddddd')

        c.node(f'train{col}', '70 % train', shape='box')
        c.node(f'model{col}', f'Model\n(BF {bf_flag})', shape='ellipse')
        c.node(f'test{col}',  '30 % test',  shape='box')

        c.edges([
            (f'data{col}', f'train{col}'),
            (f'train{col}', f'model{col}'),
            (f'model{col}', f'test{col}')
        ])

method(1, '70 % c0 → test on all cats', '—')
method(2, '70 % all cats (BF out)',     '❌')
method(3, '70 % all cats (BF in)',      '✅')

g.render()
