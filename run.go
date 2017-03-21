package main

import (
	"flag"
	"fmt"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
)

func RunCmd(args []string) {
	var netFile string
	var inFile string
	var persistent bool

	flags := flag.NewFlagSet("run", flag.ExitOnError)
	flags.StringVar(&netFile, "net", "", "input network file")
	flags.StringVar(&inFile, "in", "", "input file (stdin used by default)")
	flags.BoolVar(&persistent, "persistent", false, "persist RNN state until an empty line")
	flags.Parse(args)

	if netFile == "" {
		essentials.Die("Missing -net flag. See -help for more.")
	}

	var net *Network
	if err := serializer.LoadAny(netFile, &net); err != nil {
		essentials.Die("load network:", err)
	}

	reader, err := NewVecReaderFile(inFile)
	if err != nil {
		essentials.Die("open input:", err)
	}

	c := anyvec32.CurrentCreator()
	switch net.Net.(type) {
	case anynet.Layer:
		runFeedForward(c, net, reader)
	case anyrnn.Block:
		runRNN(c, net, reader, persistent)
	case *Bidir:
		runBidir(c, net, reader)
	case *Seq2Vec:
		runSeq2Vec(c, net, reader)
	default:
		essentials.Die("unknown network type")
	}
}

func runFeedForward(c anyvec.Creator, n *Network, r *VecReader) {
	layer := n.Net.(anynet.Layer)
	for vec := range r.InputChan() {
		if len(vec) != n.InVecSize {
			essentials.Die("bad input: expected", n.InVecSize, "inputs but got", len(vec))
		}
		inVec := c.MakeVectorData(c.MakeNumericList(vec))
		out := layer.Apply(anydiff.NewConst(inVec), 1).Output().Data()
		fmt.Println(VecStr(out))
	}
}

func runRNN(c anyvec.Creator, n *Network, r *VecReader, persist bool) {
	block := n.Net.(anyrnn.Block)
	var state anyrnn.State
	for vec := range r.InputChan() {
		if len(vec) == 0 {
			state = nil
			continue
		}
		if state == nil {
			state = block.Start(1)
		}
		if len(vec)%n.InVecSize != 0 {
			essentials.Die("bad input: length must be divisible by", n.InVecSize)
		}
		comps := splitSeq(c, vec, n.InVecSize)
		for i, comp := range comps {
			out := block.Step(state, comp)
			state = out.State()
			if i > 0 {
				fmt.Print(" ")
			}
			fmt.Print(VecStr(out.Output().Data()))
		}
		fmt.Println()
		if !persist {
			state = nil
		}
	}
}

func runBidir(c anyvec.Creator, n *Network, r *VecReader) {
	bd := n.Net.(*Bidir)
	for vec := range r.InputChan() {
		if len(vec)%n.InVecSize != 0 {
			essentials.Die("bad input: length must be divisible by", n.InVecSize)
		}
		comps := splitSeq(c, vec, n.InVecSize)
		seq := anyseq.ConstSeqList(c, [][]anyvec.Vector{comps})
		out := bd.Apply(seq).Output()
		for i, out := range out {
			if i > 0 {
				fmt.Print(" ")
			}
			fmt.Print(VecStr(out.Packed.Data()))
		}
		fmt.Println()
	}
}

func runSeq2Vec(c anyvec.Creator, n *Network, r *VecReader) {
	s2v := n.Net.(*Seq2Vec)
	for vec := range r.InputChan() {
		if len(vec)%n.InVecSize != 0 {
			essentials.Die("bad input: length must be divisible by", n.InVecSize)
		}
		comps := splitSeq(c, vec, n.InVecSize)
		seq := anyseq.ConstSeqList(c, [][]anyvec.Vector{comps})
		out := s2v.Apply(seq).Output()
		fmt.Println(VecStr(out.Data()))
	}
}
