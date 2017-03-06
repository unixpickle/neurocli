package main

import (
	"flag"
	"fmt"

	"github.com/unixpickle/essentials"
)

func SignBitCmd(args []string) {
	var inFile string

	flags := flag.NewFlagSet("signbit", flag.ExitOnError)
	flags.StringVar(&inFile, "in", "", "input file (defaults to stdin)")
	flags.Parse(args)

	reader, err := NewVecReaderFile(inFile)
	if err != nil {
		essentials.Die("open input:", err)
	}

	for vec := range reader.InputChan() {
		for i, x := range vec {
			if x >= 0 {
				vec[i] = 1
			} else {
				vec[i] = 0
			}
		}
		fmt.Println(VecStr(vec))
	}
}

func MaxCmd(args []string) {
	var inFile string
	var oneHot bool
	var vecSize int

	flags := flag.NewFlagSet("max", flag.ExitOnError)
	flags.StringVar(&inFile, "in", "", "input file (defaults to stdin)")
	flags.BoolVar(&oneHot, "onehot", false, "output one-hot vector instead of index")
	flags.IntVar(&vecSize, "vecsize", 0, "size per vector (defaults to full line)")
	flags.Parse(args)

	reader, err := NewVecReaderFile(inFile)
	if err != nil {
		essentials.Die("open input:", err)
	}

	for vec := range reader.InputChan() {
		if len(vec) == 0 {
			continue
		}
		chunk := len(vec)
		if vecSize != 0 {
			chunk = vecSize
			if len(vec)%vecSize != 0 {
				essentials.Die("bad input: length not divisible by", vecSize)
			}
		}
		for i := 0; i < len(vec); i += chunk {
			subVec := vec[i : i+chunk]
			maxIdx := 0
			maxVal := subVec[0]
			for j, x := range subVec {
				if x > maxVal {
					maxVal = x
					maxIdx = j
				}
			}
			if i != 0 {
				fmt.Print(" ")
			}
			if oneHot {
				ohVec := make([]float64, len(subVec))
				ohVec[maxIdx] = 1
				fmt.Print(VecStr(ohVec))
			} else {
				fmt.Printf("%v", maxIdx)
			}
		}
		fmt.Println()
	}
}
