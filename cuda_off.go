// +build !cuda

package main

import "github.com/unixpickle/essentials"

func enableCUDA() {
	essentials.Die("not build with CUDA support")
}
