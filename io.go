package main

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/unixpickle/essentials"
)

// VecStr turns a vector into a string.
//
// The vector may be []float64 or []float32.
func VecStr(vec interface{}) string {
	var strs []string
	switch vec := vec.(type) {
	case []float32:
		for _, x := range vec {
			strs = append(strs, fmt.Sprintf("%v", x))
		}
	case []float64:
		for _, x := range vec {
			strs = append(strs, fmt.Sprintf("%v", x))
		}
	default:
		panic("unsupported type")
	}
	return strings.Join(strs, " ")
}

// A VecReader reads space-delimited newline-separated
// vectors from an io.Reader.
type VecReader struct {
	raw io.Reader
	buf *bufio.Reader
}

// NewVecReader creates a VecReader for the reader.
func NewVecReader(r io.Reader) *VecReader {
	return &VecReader{
		raw: r,
		buf: bufio.NewReader(r),
	}
}

// NewVecReaderFile creates a VecReader from a file.
// If the filename is "", then stdin is used.
func NewVecReaderFile(file string) (*VecReader, error) {
	if file == "" {
		return NewVecReader(os.Stdin), nil
	}
	f, err := os.Open(file)
	if err != nil {
		return nil, err
	}
	return NewVecReader(f), nil
}

// Restart attempts to seek to the beginning of the input.
// If the input cannot be seeked (e.g. for stdin), this
// will return an error.
func (v *VecReader) Restart() error {
	s, ok := v.raw.(io.Seeker)
	if !ok {
		return errors.New("end of unseekable stream")
	}
	if _, err := s.Seek(0, io.SeekStart); err != nil {
		return err
	}
	v.buf.Reset(v.raw)
	return nil
}

// Read reads the next vector.
func (v *VecReader) Read() ([]float64, error) {
	line, err := v.buf.ReadString('\n')
	if err != nil {
		return nil, essentials.AddCtx("read vector", err)
	}
	comps := strings.Fields(line)
	res := make([]float64, len(comps))
	for i, c := range comps {
		var err error
		res[i], err = strconv.ParseFloat(c, 64)
		if err != nil {
			return nil, essentials.AddCtx("read vector", err)
		}
	}
	return res, nil
}

// ReadSamples reads 2*n vectors.
//
// Restart() is called if the first of a pair of vectors
// cannot be read.
// This way, inputs/outputs don't get switched for broken
// files with an odd number of lines.
//
// If an unrecoverable error is encountered, the partial
// batch is returned along with the error.
func (v *VecReader) ReadSamples(n int) (ins, outs [][]float64, err error) {
	for i := 0; i < n; i++ {
		inVec, err := v.Read()
		if err != nil {
			if err := v.Restart(); err != nil {
				return ins, outs, err
			}
			inVec, err = v.Read()
			if err != nil {
				return ins, outs, err
			}
		}
		outVec, err := v.Read()
		if err != nil {
			return ins, outs, err
		}
		ins = append(ins, inVec)
		outs = append(outs, outVec)
	}
	return
}

// InputChan produces a channel of vectors which is closed
// when reading finishes.
//
// If a non-EOF error is encountered, the program is
// terminated with an error.
func (v *VecReader) InputChan() <-chan []float64 {
	res := make(chan []float64, 1)
	go func() {
		defer close(res)
		for {
			vec, err := v.Read()
			if err != nil {
				if ctx, ok := err.(*essentials.CtxError); !ok || ctx.Original != io.EOF {
					essentials.Die(err)
				}
				return
			}
			res <- vec
		}
	}()
	return res
}
