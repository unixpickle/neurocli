package main

import (
	"errors"
	"flag"
	"io/ioutil"

	"github.com/unixpickle/anynet/anyconv"
	"github.com/unixpickle/anynet/anyrnn"
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
	block, err := parsed.Block(convmarkup.Dims{}, markupCreators())
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

	var inputCount int
	d := root.Children[0].OutDims()
	inputCount = d.Depth * d.Width * d.Height

	c := anyvec32.CurrentCreator()
	conv, err := anyconv.FromMarkupBlock(c, b, convmarkup.Dims{})
	if err == nil {
		return &Network{
			InVecSize: inputCount,
			Net:       conv,
		}
	}

	rnn, err := blockToRNN(b, &d)
	if err != nil {
		essentials.Die(err)
	}
	return &Network{
		InVecSize: inputCount,
		Net:       rnn,
	}
}

func markupCreators() map[string]convmarkup.Creator {
	res := convmarkup.DefaultCreators()
	res["LSTM"] = createMarkupLSTM
	return res
}

func blockToRNN(b convmarkup.Block, dim *convmarkup.Dims) (anyrnn.Block, error) {
	defer func() {
		*dim = b.OutDims()
	}()
	switch b := b.(type) {
	case *convmarkup.Input:
		return nil, nil
	case *convmarkup.Root:
		return blocksToRNNStack(b.Children, dim)
	case *markupLSTM:
		inSize := (*dim).Width * (*dim).Height * (*dim).Depth
		return anyrnn.NewLSTM(anyvec32.CurrentCreator(), inSize, b.StateSize), nil
	case *convmarkup.Repeat:
		var res anyrnn.Stack
		for i := 0; i < b.N; i++ {
			stack, err := blocksToRNNStack(b.Children, dim)
			if err != nil {
				return nil, err
			}
			res = append(res, stack...)
		}
		return res, nil
	default:
		layer, err := anyconv.FromMarkupBlock(anyvec32.CurrentCreator(), b, *dim)
		if err != nil {
			return nil, err
		}
		return &anyrnn.LayerBlock{Layer: layer}, err
	}
}

func blocksToRNNStack(blocks []convmarkup.Block, dim *convmarkup.Dims) (anyrnn.Stack, error) {
	var res anyrnn.Stack
	for _, b := range blocks {
		rnn, err := blockToRNN(b, dim)
		if err != nil {
			return nil, err
		} else if rnn == nil {
			continue
		}
		if stack, ok := rnn.(anyrnn.Stack); ok {
			res = append(res, stack...)
		} else {
			res = append(res, rnn)
		}
	}
	return res, nil
}

type markupLSTM struct {
	StateSize int
}

func createMarkupLSTM(in convmarkup.Dims, attr map[string]float64,
	children []convmarkup.Block) (convmarkup.Block, error) {
	if len(children) != 0 {
		return nil, errors.New("no children expected")
	}
	for x := range attr {
		if x != "out" {
			return nil, errors.New("unknown attribute: " + x)
		}
	}
	out := attr["out"]
	if out <= 0 || float64(int(out)) != out {
		return nil, errors.New("invalid or missing 'out' attribute")
	}
	return &markupLSTM{StateSize: int(out)}, nil
}

func (m *markupLSTM) Type() string {
	return "LSTM"
}

func (m *markupLSTM) OutDims() convmarkup.Dims {
	return convmarkup.Dims{Width: 1, Height: 1, Depth: m.StateSize}
}
