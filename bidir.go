package main

import (
	"flag"
	"fmt"

	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
)

func init() {
	serializer.RegisterTypedDeserializer((&Bidir{}).SerializerType(), DeserializeBidir)
}

func BidirCmd(args []string) {
	flags := flag.NewFlagSet("bidir", flag.ExitOnError)

	var forwardPath string
	var backwardPath string
	var mixerPath string
	var outPath string

	flags.StringVar(&forwardPath, "forward", "", "forward RNN file")
	flags.StringVar(&backwardPath, "backward", "", "backward RNN file")
	flags.StringVar(&mixerPath, "mixer", "", "mixer feed-forward network file")
	flags.StringVar(&outPath, "out", "", "output file")
	flags.Parse(args)

	if forwardPath == "" || backwardPath == "" || mixerPath == "" || outPath == "" {
		essentials.Die("Required flags: -forward, -backward, -mixer, -out.\n" +
			"See -help for more.")
	}

	var forward, backward, mixer *Network
	ptrs := []**Network{&forward, &backward, &mixer}
	paths := []string{forwardPath, backwardPath, mixerPath}
	names := []string{"forward", "backward", "mixer"}
	for i, ptr := range ptrs {
		if err := serializer.LoadAny(paths[i], ptr); err != nil {
			essentials.Die("load "+names[i]+":", err)
		}
		if i < 2 {
			if !(*ptr).RNN() {
				essentials.Die("expected", names[i], "to be an RNN")
			}
		} else {
			if !(*ptr).FeedForward() {
				essentials.Die("expected", names[i], "to be a feed-forward network.")
			}
		}
	}

	if mixer.InVecSize != forward.OutVecSize+backward.OutVecSize {
		essentials.Die(fmt.Sprintf("mixer input size should be %d (not %d)",
			forward.OutVecSize+backward.OutVecSize, mixer.InVecSize))
	} else if forward.InVecSize != backward.InVecSize {
		essentials.Die("mismatched forward and backward input sizes")
	}

	net := &Network{
		InVecSize:  forward.InVecSize,
		OutVecSize: mixer.OutVecSize,
		Net: &Bidir{
			In: &anyrnn.Bidir{
				Forward:  forward.Net.(anyrnn.Block),
				Backward: backward.Net.(anyrnn.Block),
				Mixer:    anynet.ConcatMixer{},
			},
			Out: mixer.Net.(anynet.Layer),
		},
	}

	if err := serializer.SaveAny(outPath, net); err != nil {
		essentials.Die(err)
	}
}

// Bidir is a bidirectional RNN with an included output
// layer.
type Bidir struct {
	In  *anyrnn.Bidir
	Out anynet.Layer
}

// DeserializeBidir deserializes a Bidir.
func DeserializeBidir(d []byte) (*Bidir, error) {
	var b Bidir
	if err := serializer.DeserializeAny(d, &b.In, &b.Out); err != nil {
		return nil, err
	}
	return &b, nil
}

// Apply applies the bidirectional RNN to a sequence.
func (b *Bidir) Apply(in anyseq.Seq) anyseq.Seq {
	return anyseq.Map(b.In.Apply(in), b.Out.Apply)
}

// SerializerType returns the unique ID used to serialize
// a Bidir with the serializer package.
func (b *Bidir) SerializerType() string {
	return "github.com/unixpickle/neurocli.Bidir"
}

// Serialize serializes the Bidir.
func (b *Bidir) Serialize() ([]byte, error) {
	return serializer.SerializeAny(b.In, b.Out)
}
