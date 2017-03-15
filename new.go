package main

import (
	"flag"
	"fmt"
	"io/ioutil"

	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyconv"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/convmarkup"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
)

func NewCmd(args []string) {
	flags := flag.NewFlagSet("new", flag.ExitOnError)

	var markupFile string
	var outputFile string

	flags.StringVar(&markupFile, "in", "", "network markup file")
	flags.StringVar(&outputFile, "out", "", "output network file")
	flags.Parse(args)

	if markupFile == "" || outputFile == "" {
		essentials.Die("Required flags: -in and -out. See -help for more.")
	}

	markup, err := ioutil.ReadFile(markupFile)
	if err != nil {
		essentials.Die(err)
	}

	parsed, err := convmarkup.Parse(string(markup))
	if err != nil {
		essentials.Die(err)
	}
	block, err := parsed.Block(convmarkup.Dims{}, anyrnn.MarkupCreators())
	if err != nil {
		essentials.Die(err)
	}

	net := networkFromBlock(block)
	if err := serializer.SaveAny(outputFile, net); err != nil {
		essentials.Die(err)
	}
}

func networkFromBlock(b convmarkup.Block) *Network {
	root := b.(*convmarkup.Root)
	if len(root.Children) == 0 {
		essentials.Die("no blocks in markup file")
	}

	inputCount := root.Children[0].OutDims().Volume()
	outputCount := root.OutDims().Volume()

	c := anyvec32.CurrentCreator()
	conv, err := convFromMarkup(c, convmarkup.Dims{}, b)
	if err == nil {
		return &Network{
			InVecSize:  inputCount,
			OutVecSize: outputCount,
			Net:        conv,
		}
	}
	rnn, err := rnnFromMarkup(c, b)
	if err != nil {
		essentials.Die(err)
	}
	return &Network{
		InVecSize:  inputCount,
		OutVecSize: outputCount,
		Net:        rnn,
	}
}

func convFromMarkup(c anyvec.Creator, inDims convmarkup.Dims,
	b convmarkup.Block) (anynet.Layer, error) {
	chain := convmarkup.RealizerChain{
		convmarkup.MetaRealizer{},
		anyconv.Realizer(anyvec32.CurrentCreator()),
	}
	instance, _, err := chain.Realize(inDims, b)
	if err != nil {
		return nil, err
	}
	if layer, ok := instance.(anynet.Layer); ok {
		return layer, nil
	} else {
		return nil, fmt.Errorf("not an anynet.Layer: %T", instance)
	}
}

func rnnFromMarkup(c anyvec.Creator, b convmarkup.Block) (anyrnn.Block, error) {
	chain := convmarkup.RealizerChain{
		convmarkup.MetaRealizer{},
		anyrnn.Realizer(c, convmarkup.RealizerChain{
			convmarkup.MetaRealizer{},
			anyconv.Realizer(c),
		}),
	}
	instance, _, err := chain.Realize(convmarkup.Dims{}, b)
	if err != nil {
		return nil, err
	}
	if block, ok := instance.(anyrnn.Block); ok {
		return block, nil
	} else {
		return nil, fmt.Errorf("not an anyrnn.Block: %T", instance)
	}
}
