#! /usr/bin/python
#
# A little python code to read and process MINERvA's
# flux prediction. Adapt to your taste.
# Orignal development with python 2.7.9, ROOT 5.34/28
#
# Author: Mike Kordosky, makordosky@wm.edu
#
import glob, os, re, sys, getopt, array

# from ROOT import TPad,TCanvas,TStyle,TMath
from ROOT import *


def read_flux_file(name):
    f = open(name)
    cntr = 1
    lines = []
    # read in each line, split by whitespace, convert to floats
    # and store in an array. skip blank lines
    # stick the arrays into a list
    for line in f:
        v = array.array("f", [float(val) for val in line.split()])
        n = len(v)
        if n == 0:
            continue
        cntr += 1
        lines.append(v)
    print(len(lines))

    # now the first element is the flux
    flux = lines.pop(0)
    print("The flux (nu/m^2/POT) in 0.5 GeV bins:")
    print(flux)
    # the second is the uncertainty
    flux_unc = lines.pop(0)
    print("The uncertainty (nu/m^2/POT) in 0.5 GeV bins:")
    print(flux_unc)
    # the rest makeup the covariance matrix
    if len(lines) != len(lines[0]):
        print("Houston we have a problem!")
        print("the file ", name, " doesn't have a square covariance matrix")
    print("Flux file has a ", len(lines), "x", len(lines[0]), " covariance matrix")
    return flux, flux_unc, lines


def make_flux_histogram(flux, flux_unc, name="flux", title=";neutrino energy (GeV); #nu/m^{2}/10^{6}POT/0.5 GeV", emax=100, emin=0):
    h = TH1F(name, title, len(flux), emin, emax)
    for val, err, i in zip(flux, flux_unc, range(1, len(flux) + 1)):
        h.SetBinContent(i, val * 1e6)
        h.SetBinError(i, err * 1e6)
    return h


def make_covmx_histogram(covmx, name="covmx", title=";neutrino energy bin; neutrino energy bin"):
    n = len(covmx)
    h = TH2F(name, title, n, 1, n, n, 1, n)
    for row, i in zip(covmx, range(1, n + 1)):
        for val, j in zip(row, range(1, n + 1)):
            h.SetBinContent(i, j, val * 1e6)
    return h


def make_cormx_histogram(covmx, name="cormx", title=";neutrino energy bin; neutrino energy bin"):
    n = len(covmx)
    h = TH2F(name, title, n, 1, n, n, 1, n)
    sigmas = []
    for row, i in zip(covmx, range(0, n)):
        sigmas.append(sqrt(row[i]))

    for row, i in zip(covmx, range(1, n + 1)):
        for val, j in zip(row, range(1, n + 1)):
            sigi = sigmas[i - 1]  # root histos start at bin 1, arrays at index 0
            sigj = sigmas[j - 1]
            if sigi == 0 or sigj == 0:
                h.SetBinContent(i, j, 0)
            else:
                h.SetBinContent(i, j, val / (sigi * sigj))
    return h


def make_flux_table(flux, unc, caption="hello", highbin=200, lowbin=0):
    header1 = "\\begin{longtable}{|c|cc||c|cc|}\n"
    capstr = "\\caption{{ {:s} }}\\\ \n".format(caption)
    header2 = r"""\hline
E (GeV) & $\phi$ & $\delta \phi$ (\%) & E (GeV) & $\phi$ & $\delta \phi$ (\%) \endhead
\hline
"""
    footer = r" \end{longtable}"

    output = [header1]
    output.append(capstr)
    output.append(header2)
    binw = 0.5
    for i in range(lowbin, highbin, 2):
        ehigh1 = binw + i * binw
        elow1 = i * binw
        ehigh2 = binw + (i + 1) * binw
        elow2 = (i + 1) * binw
        pctunc1 = 0.0 if flux[i] == 0 else unc[i] / flux[i] * 100
        pctunc2 = 0.0 if flux[i + 1] == 0 else unc[i + 1] / flux[i + 1] * 100
        s = r"""{}-{} & {:2.2e} & {:1.1f} & {}-{} & {:2.2e} & {:1.1f}\\
\hline""".format(
            elow1, ehigh1, flux[i] * 1e6, pctunc1, elow2, ehigh2, flux[i + 1] * 1e6, pctunc2
        )
        output.append(s)
    output.append(footer)
    return output


def write_table_to_file(table, filename):
    f = open(filename, "w")
    for line in table:
        f.write(line)
    f.close()


c1 = TCanvas()
flux, unc, covmx = read_flux_file("Nu_Flux_wandcsplinefix_numuFHC.txt")
h = make_flux_histogram(flux, unc, "numu_fhc")
h.Draw()

c2 = TCanvas()
hcovmx = make_covmx_histogram(covmx, "numu_fhc_covmx")
hcovmx.Draw("colz")

c3 = TCanvas()
hcormx = make_cormx_histogram(covmx, "numu_fhc_cormx")
hcormx.Draw("colz")

cap = r"The \numu flux in units of $\nu/m^2/10^6 POT$ for the FHC beam."
table = make_flux_table(flux, unc, cap, 50)

write_table_to_file(table, "junk.tex")

# input_template="Nu_Flux_wandcsplinefix_{:s}{:s}_constrained.txt"
input_template = "{:s}{:s}.txt"
histo_template = "{:s}_{:s}"
covmx_template = "{:s}_{:s}_covmx"
cormx_template = "{:s}_{:s}_cormx"
caption_template = r"The \{:s} flux in units of $\nu/m^2/10^6 POT$ for the {:s} beam."

neutrinos = ["numu", "numubar", "nue", "nuebar"]
beams = ["FHC", "RHC"]
# neutrinos=["numu","numubar"]
# beams=["FHC"]
outfile = TFile("minerva_flux.root", "recreate")
maxenergy = 20
maxbin = 40
gStyle.SetOptStat(0)
for beam in beams:
    for neutrino in neutrinos:
        infile = input_template.format(neutrino, beam)
        histo_name = histo_template.format(neutrino, beam.swapcase())
        tex_name = histo_name + ".tex"
        histo_pdf = histo_name + ".pdf"
        covmx_name = covmx_template.format(neutrino, beam.swapcase())
        cormx_name = cormx_template.format(neutrino, beam.swapcase())
        cormx_pdf = cormx_name + ".pdf"
        caption = caption_template.format(neutrino, beam)
        print(infile)
        print(histo_name)
        print(tex_name)
        print(histo_pdf)
        print(covmx_name)
        print(cormx_name)
        print(cormx_pdf)
        flux, unc, covmx = read_flux_file(infile)
        h = make_flux_histogram(flux, unc, histo_name)
        hcovmx = make_covmx_histogram(covmx, covmx_name)
        hcormx = make_cormx_histogram(covmx, cormx_name)
        table = make_flux_table(flux, unc, caption, maxbin)
        write_table_to_file(table, tex_name)
        c1.cd()
        h.SetFillColor(kGray)
        hcopy = h.DrawCopy("e2")
        h.SetFillStyle(0)
        h.Draw("hist same")

        hcopy.GetXaxis().SetRangeUser(0, maxenergy)
        c1.Print(histo_pdf)
        c2.cd()
        hcormx.Draw("colz")
        hcormx.GetXaxis().SetRangeUser(0, maxbin)
        hcormx.GetYaxis().SetRangeUser(0, maxbin)
        c2.Print(cormx_pdf)
        h.SetDirectory(outfile)
        hcovmx.SetDirectory(outfile)
        hcormx.SetDirectory(outfile)
        h.Write()
        hcovmx.Write()
        hcormx.Write()

TPython.Prompt()
