# A library and utility for drawing ONNX nets. Most of this implementation has
# been borrowed from the caffe2 implementation
# https://github.com/caffe2/caffe2/blob/master/caffe2/python/net_drawer.py
#
# The script takes two required arguments:
#   -input: a path to a serialized ModelProto .pb file.
#   -output: a path to write a dot file representation of the graph
#
# Given this dot file representation, you can-for example-export this to svg
# with the graphviz `dot` utility, like so:
#
#   $ dot -Tsvg my_output.dot -o my_output.svg
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
from collections import defaultdict
import json
from onnx import ModelProto
import pydot


OP_STYLE = {
    'shape': 'box',
    'color': '#0F9D58',
    'style': 'filled',
    'fontcolor': '#FFFFFF'
}

BLOB_STYLE = {'shape': 'octagon'}


def _escape_label(name):
    # json.dumps is poor man's escaping
    return json.dumps(name)


def _form_and_sanitize_docstring(s):
    url = 'javascript:alert('
    url += _escape_label(s).replace('"', '\'').replace('<', '').replace('>', '')
    url += ')'
    return url

number_add = 0
number_mul = 0
number_conv = 0
number_batch = 0

row = 2
column = 2
distance = 20

def batchNormalization(op_style):
    global number_batch
    number_batch += 1
    if number_batch == 1:
        op_style['pos'] = str(0) + "," + str(0) + "!"
    if number_batch == (2 + ((row-1)*column + (column-1)*row)*2 + 2):
        op_style['pos'] = str((row-1+2)*distance) + "," + str(0) + "!"
    if number_batch == 3:
        op_style['pos'] = str(distance) + "," + str(0) + "!"
    op_style['color'] = '#0F9D58'
    return op_style


def mul(op_style):
    global number_mul
    number_mul += 1
    op_style['color'] = '#d84d0d'
    return op_style


def add(op_style):
    global number_add
    number_add += 1
    if number_add == 1:
        op_style['pos'] = str(distance*2) + "," + str(distance) + "!"
    if number_add == 2:
        op_style['pos'] = str(distance*2) + "," + str(0) + "!"
    op_style['color'] = '#c119a8'
    return op_style


def conv(op_style):
    global number_conv
    number_conv += 1
    op_style['color'] = '#266570'
    return op_style


def constant(op_style):
    op_style['color'] = '#fff200'
    return op_style


def relu(op_style):
    op_style['color'] = '#0021ff'
    return op_style

style = {"BatchNormalization": batchNormalization,
         "Mul": mul,
         "Add": add,
         "Conv": conv,
         "Constant": constant,
         "Relu": relu,
         }

def GetOpNodeProducer(embed_docstring=False, **kwargs):
    def ReallyGetOpNode(op, op_id):
        if op.name:
            node_name = '%s/%s (op#%d)' % (op.name, op.op_type, op_id)
        else:
            node_name = '%s (op#%d)' % (op.op_type, op_id)
        for i, input in enumerate(op.input):
            node_name += '\n input' + str(i) + ' ' + input

        for i, output in enumerate(op.output):
            node_name += '\n output' + str(i) + ' ' + output

        OP_STYLE_init = {
            'shape': 'box',
            'color': '#0F9D58',
            'style': 'filled',
            'fontcolor': '#FFFFFF',
        }

        OP_STYLE_final = style[op.op_type](OP_STYLE_init)

        node = pydot.Node(node_name, **OP_STYLE_final)


        if embed_docstring:
            url = _form_and_sanitize_docstring(op.doc_string)
            node.set_URL(url)
        return node

    return ReallyGetOpNode


def GetPydotGraph(graph,
                  name=None,
                  rankdir='LR',
                  node_producer=None,
                  embed_docstring=False):

    if node_producer is None:
        node_producer = GetOpNodeProducer(embed_docstring=embed_docstring, **OP_STYLE)
    pydot_graph = pydot.Dot(name, rankdir=rankdir)
    pydot_nodes = {}
    pydot_node_counts = defaultdict(int)
    for op_id, op in enumerate(graph.node):
        op_node = node_producer(op, op_id)

        pydot_graph.add_node(op_node)

        #print(pydot_graph)
        for input_name in op.input:
            if input_name not in pydot_nodes:
                input_node = pydot.Node(_escape_label(input_name + str(pydot_node_counts[input_name])),
                                        label=_escape_label(input_name),
                                        **BLOB_STYLE
                                        )
                pydot_nodes[input_name] = input_node
            else:
                input_node = pydot_nodes[input_name]
            pydot_graph.add_node(input_node)
            pydot_graph.add_edge(pydot.Edge(input_node, op_node))
        #print("graph")
        #print(pydot_graph)

        for output_name in op.output:
            if output_name in pydot_nodes:
                pydot_node_counts[output_name] += 1
            output_node = pydot.Node(_escape_label(output_name + str(pydot_node_counts[output_name])),
                                     label=_escape_label(output_name),
                                     **BLOB_STYLE
                                     )
            pydot_nodes[output_name] = output_node
            pydot_graph.add_node(output_node)
            pydot_graph.add_edge(pydot.Edge(op_node, output_node))

    return pydot_graph


def main():
    parser = argparse.ArgumentParser(description="ONNX net drawer")
    print("Start")
    parser.add_argument(
        "--input",
        type=str, required=True,
        help="The input protobuf file.",
    )
    parser.add_argument(
        "--output",
        type=str, required=True,
        help="The output protobuf file.",
    )
    parser.add_argument(
        "--rankdir", type=str, default='LR',
        help="The rank direction of the pydot graph.",
    )
    parser.add_argument(
        "--embed_docstring", action="store_true",
        help="Embed docstring as javascript alert. Useful for SVG format.",
    )
    args = parser.parse_args()
    model = ModelProto()
    with open(args.input, 'rb') as fid:
        content = fid.read()
        model.ParseFromString(content)
    pydot_graph = GetPydotGraph(
            model.graph,
            name=model.graph.name,
            rankdir=args.rankdir,
            node_producer=GetOpNodeProducer(
                embed_docstring=args.embed_docstring,
                **OP_STYLE
        ),
    )
    with open("./GridNet_s.dot", 'w') as txtfile:
        txtfile.write(pydot_graph.to_string())
    pydot_graph.write_dot(args.output)


if __name__ == '__main__':
    main()
