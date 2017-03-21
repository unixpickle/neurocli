package main

import (
	"fmt"
	"os"
)

var (
	Commands = []string{
		"new", "train", "run", "signbit", "max", "read", "write",
		"bidir", "seq2vec",
	}
	CommandDescs = map[string]string{
		"new":     "create a new network file",
		"train":   "train a network on data",
		"run":     "run new samples through a network",
		"signbit": "compute the sign bit of vector components",
		"max":     "get the index of the max vector component",
		"read":    "read the parameters of a network",
		"write":   "write the parameters of a network",
		"bidir":   "create bidirectional RNN",
		"seq2vec": "create sequence-to-vector model",
	}
	CommandFuncs = map[string]func([]string){
		"new":     NewCmd,
		"train":   TrainCmd,
		"run":     RunCmd,
		"signbit": SignBitCmd,
		"max":     MaxCmd,
		"read":    ReadCmd,
		"write":   WriteCmd,
		"bidir":   BidirCmd,
		"seq2vec": Seq2VecCmd,
	}
)

func main() {
	if len(os.Args) < 2 {
		dieUsage()
	}
	cmd := os.Args[1]
	if f, ok := CommandFuncs[cmd]; ok {
		f(os.Args[2:])
	} else {
		fmt.Fprintln(os.Stderr, "Unknown command:", cmd)
		dieUsage()
	}
}

func dieUsage() {
	fmt.Fprintln(os.Stderr, "Usage: neurocli <command> [flags | -help]")
	fmt.Fprintln(os.Stderr)
	fmt.Fprintln(os.Stderr, "Available commands:")
	for _, name := range Commands {
		desc := CommandDescs[name]
		for len(name) < 8 {
			name += " "
		}
		fmt.Fprintln(os.Stderr, name+desc)
	}
	fmt.Fprintln(os.Stderr)
	os.Exit(1)
}
