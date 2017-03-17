package main

import (
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/serializer"
)

func init() {
	serializer.RegisterTypedDeserializer((&Network{}).SerializerType(), DeserializeNetwork)
}

// A Network contains a network and its metadata.
type Network struct {
	InVecSize  int
	OutVecSize int

	Net interface{}
}

// DeserializeNetwork deserializes a Network.
func DeserializeNetwork(d []byte) (*Network, error) {
	var res Network
	err := serializer.DeserializeAny(d, &res.InVecSize, &res.OutVecSize, &res.Net)
	if err != nil {
		return nil, err
	}
	return &res, nil
}

// FeedForward returns true if the network is a regular
// feed-forward neural network (rather than an RNN).
func (n *Network) FeedForward() bool {
	_, ok := n.Net.(anynet.Layer)
	return ok
}

// RNN returns true if the network is a regular
// unidirectional RNN.
func (n *Network) RNN() bool {
	_, ok := n.Net.(anyrnn.Block)
	return ok
}

// Bidir returns true if the network is a bidirectional
// RNN.
func (n *Network) Bidir() bool {
	_, ok := n.Net.(*Bidir)
	return ok
}

// SerializerType returns the unique ID used to serialize
// a Network with the serializer package.
func (n *Network) SerializerType() string {
	return "github.com/unixpickle/neurocli.Network"
}

// Serialize serializes the Network.
func (n *Network) Serialize() ([]byte, error) {
	return serializer.SerializeAny(n.InVecSize, n.OutVecSize, n.Net)
}
