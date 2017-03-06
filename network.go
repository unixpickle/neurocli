package main

import (
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/serializer"
)

func init() {
	serializer.RegisterTypedDeserializer((&Network{}).SerializerType(), DeserializeNetwork)
}

// A Network contains a network and its metadata.
type Network struct {
	InVecSize int
	Net       interface{}
}

// DeserializeNetwork deserializes a Network.
func DeserializeNetwork(d []byte) (*Network, error) {
	var res Network
	if err := serializer.DeserializeAny(d, &res.InVecSize, &res.Net); err != nil {
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

// SerializerType returns the unique ID used to serialize
// a Network with the serializer package.
func (n *Network) SerializerType() string {
	return "github.com/unixpickle/neurocli.Network"
}

// Serialize serializes the Network.
func (n *Network) Serialize() ([]byte, error) {
	return serializer.SerializeAny(n.InVecSize, n.Net)
}
