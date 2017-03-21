package main

import (
	"flag"
	"fmt"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
)

func init() {
	serializer.RegisterTypedDeserializer((&Seq2Vec{}).SerializerType(), DeserializeSeq2Vec)
}

func Seq2VecCmd(args []string) {
	flags := flag.NewFlagSet("seq2vec", flag.ExitOnError)

	var rnnPath string
	var outNetPath string
	var outPath string

	flags.StringVar(&rnnPath, "rnn", "", "RNN file")
	flags.StringVar(&outNetPath, "outnet", "", "feed-forward network file")
	flags.StringVar(&outPath, "out", "", "output file")
	flags.Parse(args)

	if rnnPath == "" || outNetPath == "" || outPath == "" {
		essentials.Die("Required flags: -rnn, -outnet, and -out.\n" +
			"See -help for more.")
	}

	var rnn, outNet *Network
	if err := serializer.LoadAny(rnnPath, &rnn); err != nil {
		essentials.Die("load rnn:", err)
	}
	if _, ok := rnn.Net.(anyrnn.Block); !ok {
		essentials.Die("load rnn: unexpected network type")
	}
	if err := serializer.LoadAny(outNetPath, &outNet); err != nil {
		essentials.Die("load out net:", err)
	}
	if _, ok := outNet.Net.(anynet.Layer); !ok {
		essentials.Die("load out net: unexpected network type")
	}

	if outNet.InVecSize != rnn.OutVecSize {
		essentials.Die(fmt.Sprintf("out net input size should be %d (not %d)",
			rnn.OutVecSize, outNet.InVecSize))
	}

	net := &Network{
		InVecSize:  rnn.InVecSize,
		OutVecSize: outNet.OutVecSize,
		Net: &Seq2Vec{
			Block: rnn.Net.(anyrnn.Block),
			Out:   outNet.Net.(anynet.Layer),
		},
	}

	if err := serializer.SaveAny(outPath, net); err != nil {
		essentials.Die(err)
	}
}

// Seq2Vec is an RNN which maps input sequences to output
// vectors.
type Seq2Vec struct {
	Block anyrnn.Block
	Out   anynet.Layer
}

// DeserializeSeq2Vec deserializes a Seq2Vec.
func DeserializeSeq2Vec(d []byte) (*Seq2Vec, error) {
	var res Seq2Vec
	if err := serializer.DeserializeAny(d, &res.Block, &res.Out); err != nil {
		return nil, err
	}
	return &res, nil
}

// Apply applies the network to a sequence.
func (s *Seq2Vec) Apply(in anyseq.Seq) anydiff.Res {
	if len(in.Output()) == 0 {
		return anydiff.NewConst(in.Creator().MakeVector(0))
	}
	tail := anyseq.Tail(anyrnn.Map(in, s.Block))
	n := in.Output()[0].NumPresent()
	return s.Out.Apply(tail, n)
}

// Parameters returns the network's parameters.
func (s *Seq2Vec) Parameters() []*anydiff.Var {
	return anynet.AllParameters(s.Block, s.Out)
}

// SerializerType returns the unique ID used to serialize
// a Seq2Vec with the serializer package.
func (s *Seq2Vec) SerializerType() string {
	return "github.com/unixpickle/neurocli.Seq2Vec"
}

// Serilaize serializes the Seq2Vec.
func (s *Seq2Vec) Serialize() ([]byte, error) {
	return serializer.SerializeAny(s.Block, s.Out)
}
