package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/unixpickle/anynet"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
)

func ReadCmd(args []string) {
	var netFile string

	flags := flag.NewFlagSet("read", flag.ExitOnError)
	flags.StringVar(&netFile, "net", "", "network file path")
	flags.Parse(args)

	if netFile == "" {
		essentials.Die("Missing -net flag. See -help for more.")
	}

	var net *Network
	if err := serializer.LoadAny(netFile, &net); err != nil {
		essentials.Die("load network:", err)
	}

	for _, p := range net.Net.(anynet.Parameterizer).Parameters() {
		fmt.Println(VecStr(p.Vector.Data()))
	}
}

func WriteCmd(args []string) {
	var netFile string
	var inFile string

	flags := flag.NewFlagSet("write", flag.ExitOnError)
	flags.StringVar(&netFile, "net", "", "network file path")
	flags.StringVar(&inFile, "in", "", "input file (use stdin by default)")
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

	for i, p := range net.Net.(anynet.Parameterizer).Parameters() {
		vec, err := reader.Read()
		if err != nil {
			essentials.Die(err)
		}
		if len(vec) != p.Vector.Len() {
			fmt.Fprintf(os.Stderr, "length mismatch: parameter %d is %d components but got %d",
				i, p.Vector.Len(), len(vec))
			fmt.Fprintln(os.Stderr)
			os.Exit(1)
		}
		p.Vector.SetData(p.Vector.Creator().MakeNumericList(vec))
	}

	if err := serializer.SaveAny(netFile, net); err != nil {
		essentials.Die("save network:", err)
	}
}
