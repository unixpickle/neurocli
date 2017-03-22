package main

import (
	"flag"
	"fmt"
	"strings"

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
	var batch int

	flags := flag.NewFlagSet("run", flag.ExitOnError)
	flags.StringVar(&netFile, "net", "", "input network file")
	flags.StringVar(&inFile, "in", "", "input file (stdin used by default)")
	flags.BoolVar(&persistent, "persistent", false, "persist RNN state until an empty line")
	flags.IntVar(&batch, "batch", 1, "evaluation batch size")
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
		runFeedForward(c, net, reader, batch)
	case anyrnn.Block:
		if persistent {
			if batch != 1 {
				essentials.Die("cannot combine -batch and -peristent")
			}
			runPersistentRNN(c, net, reader)
		} else {
			block := net.Net.(anyrnn.Block)
			runSeq2Seq(c, net, reader, batch, func(in anyseq.Seq) anyseq.Seq {
				return anyrnn.Map(in, block)
			})
		}
	case *Bidir:
		bd := net.Net.(*Bidir)
		runSeq2Seq(c, net, reader, batch, bd.Apply)
	case *Seq2Vec:
		runSeq2Vec(c, net, reader, batch)
	default:
		essentials.Die("unknown network type")
	}
}

func runFeedForward(c anyvec.Creator, n *Network, r *VecReader, batchSize int) {
	var batch []anyvec.Vector
	doBatch := func() {
		layer := n.Net.(anynet.Layer)
		joined := anydiff.NewConst(c.Concat(batch...))
		out := layer.Apply(joined, len(batch)).Output()
		printSplit(out, len(batch))
		batch = nil
	}
	for vec := range r.InputChan() {
		if len(vec) != n.InVecSize {
			essentials.Die("bad input: expected", n.InVecSize, "inputs but got", len(vec))
		}
		v := c.MakeVectorData(c.MakeNumericList(vec))
		batch = append(batch, v)
		if len(batch) >= batchSize {
			doBatch()
		}
	}
	if len(batch) > 0 {
		doBatch()
	}
}

func runPersistentRNN(c anyvec.Creator, n *Network, r *VecReader) {
	block := n.Net.(anyrnn.Block)
	state := block.Start(1)
	for vec := range r.InputChan() {
		if len(vec) == 0 {
			state = block.Start(1)
			continue
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
	}
}

func runSeq2Seq(c anyvec.Creator, n *Network, r *VecReader, batchSize int,
	f func(anyseq.Seq) anyseq.Seq) {
	var batch [][]anyvec.Vector
	doBatch := func() {
		seqs := anyseq.ConstSeqList(c, batch)
		printSeqs(f(seqs).Output(), len(batch))
		batch = nil
	}
	for vec := range r.InputChan() {
		if len(vec)%n.InVecSize != 0 {
			essentials.Die("bad input: length must be divisible by", n.InVecSize)
		}
		batch = append(batch, splitSeq(c, vec, n.InVecSize))
		if len(batch) >= batchSize {
			doBatch()
		}
	}
	if len(batch) > 0 {
		doBatch()
	}
}

func runSeq2Vec(c anyvec.Creator, n *Network, r *VecReader, batchSize int) {
	var batch [][]anyvec.Vector
	doBatch := func() {
		seqs := anyseq.ConstSeqList(c, batch)
		out := n.Net.(*Seq2Vec).Apply(seqs).Output()
		printSplit(out, len(batch))
		batch = nil
	}
	for vec := range r.InputChan() {
		if len(vec)%n.InVecSize != 0 {
			essentials.Die("bad input: length must be divisible by", n.InVecSize)
		} else if len(vec) == 0 {
			essentials.Die("bad input: cannot be empty")
		}
		batch = append(batch, splitSeq(c, vec, n.InVecSize))
		if len(batch) >= batchSize {
			doBatch()
		}
	}
	if len(batch) > 0 {
		doBatch()
	}
}

func printSplit(v anyvec.Vector, n int) {
	segLen := v.Len() / n
	for i := 0; i < n; i++ {
		subVec := v.Slice(i*segLen, (i+1)*segLen)
		fmt.Println(VecStr(subVec.Data()))
	}
}

func printSeqs(seq []*anyseq.Batch, n int) {
	for i := 0; i < n; i++ {
		pres := make([]bool, n)
		pres[i] = true
		var parts []string
		for _, batch := range seq {
			if !batch.Present[i] {
				break
			}
			vec := batch.Reduce(pres).Packed
			parts = append(parts, VecStr(vec.Data()))
		}
		fmt.Println(strings.Join(parts, " "))
	}
}
