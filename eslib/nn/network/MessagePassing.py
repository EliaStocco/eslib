from typing import TypeVar

import torch
from e3nn import o3
from e3nn.nn import BatchNorm, Dropout, Gate

from eslib.nn.network.Convolution import Convolution

C = TypeVar('C', bound='Compose')
T = TypeVar('T', bound='MessagePassing')
# See https://mypy.readthedocs.io/en/latest/generics.html#generic-methods-and-generic-self for the use
# of `T` to annotate `self`. Many methods of `Module` return `self` and we want those return values to be
# the type of the subclass, not the looser type of `Module`.

def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False

class Compose(torch.nn.Module):

    first : torch.nn.Module
    second : torch.nn.Module

    def __init__(self:C, first: torch.nn.Module, second: torch.nn.Module)->None:
        super().__init__()
        self.first = first
        self.second = second

        try :
            self.irreps_in = self.first.irreps_in
        except :
            self.irreps_in = self.first.irreps_node_input
        
        try :
            self.irreps_out = self.second.irreps_out
        except :
            try :
                self.irreps_out = self.second.irreps_node_output
            except : 
                self.irreps_out = self.second.irreps


    def forward(self:C, *input):
        x = self.first(*input)
        return self.second(x)

class MessagePassing(torch.nn.Module):
    r"""

    Parameters
    ----------
    irreps_node_input : `e3nn.o3.Irreps`
        representation of the input features

    irreps_node_hidden : `e3nn.o3.Irreps`
        representation of the hidden features

    irreps_node_output : `e3nn.o3.Irreps`
        representation of the output features

    irreps_node_attr : `e3nn.o3.Irreps`
        representation of the nodes attributes

    irreps_edge_attr : `e3nn.o3.Irreps`
        representation of the edge attributes

    layers : int
        number of gates (non linearities)

    fc_neurons : list of int
        number of neurons per layers in the fully connected network
        first layer and hidden layers but not the output layer
    """

    def __init__(
        self:T,
        irreps_node_input,
        irreps_node_hidden,
        irreps_node_output,
        irreps_node_attr,
        irreps_edge_attr,
        layers,
        fc_neurons,
        num_neighbors,
        debug=False,
        dropout_probability=0.,
        batchnorm=False,
    ) -> None:
        super().__init__()
        self.num_neighbors = num_neighbors

        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_hidden = o3.Irreps(irreps_node_hidden)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.debug = debug

        self.irreps_in  = self.irreps_node_input
        self.irreps_out = self.irreps_node_output
        irreps_node = self.irreps_node_input
        act = {
            1: torch.nn.functional.silu,
            -1: torch.tanh,
        }
        act_gates = {
            1: torch.sigmoid,
            -1: torch.tanh,
        }
        self.layers = torch.nn.ModuleList()
        for _ in range(layers):
            irreps_scalars = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in self.irreps_node_hidden
                    if ir.l == 0 and tp_path_exists(irreps_node, self.irreps_edge_attr, ir)
                ]
            ).simplify()
            irreps_gated = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in self.irreps_node_hidden
                    if ir.l > 0 and tp_path_exists(irreps_node, self.irreps_edge_attr, ir)
                ]
            )
            ir = "0e" if tp_path_exists(irreps_node, self.irreps_edge_attr, "0e") else "0o"
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()
            gate = Gate(
                irreps_scalars,
                [act[ir.p] for _, ir in irreps_scalars],  # scalar
                irreps_gates,
                [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated,  # gated tensors
            )
            conv = Convolution(irreps_node, self.irreps_node_attr, self.irreps_edge_attr, gate.irreps_in, fc_neurons, num_neighbors)
            irreps_node = gate.irreps_out
            if batchnorm :
                raise ValueError("not implemented yet")
                bn = BatchNorm(irreps=conv.irreps_out)
                tmp = Compose(conv, bn)
                self.layers.append(Compose(tmp, gate))
            else :
                self.layers.append(Compose(conv, gate))

        conv = Convolution(irreps_node, self.irreps_node_attr, self.irreps_edge_attr,self.irreps_out, fc_neurons, num_neighbors)
        self.layers.append(conv)

        if dropout_probability < 0 :
            raise ValueError("'dropout_probability' should be >= zero")
        elif dropout_probability > 0 :
            self.dropout = torch.nn.ModuleList()
            for lay in self.layers :
                try :
                    self.dropout.append(Dropout(irreps=lay.irreps_out,p=dropout_probability))
                except:
                    self.dropout.append(Dropout(irreps=lay.second._irreps_out,p=dropout_probability))

            self.dropout_probability = dropout_probability
        else :
            self.dropout = None
        pass

    def forward(self:T, node_features, node_attr, edge_src, edge_dst, edge_attr, edge_scalars) -> torch.Tensor:
        for n,lay in enumerate(self.layers):
            node_features = lay(node_features, node_attr, edge_src, edge_dst, edge_attr, edge_scalars)
            if self.dropout is not None :
                node_features = self.dropout[n](node_features)
        return node_features