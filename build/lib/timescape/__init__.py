__version__='1.0.1'
__author__='Christopher Harvey-Hawes'
__email__='christopher.harvey-hawes@pg.canterbury.ac.nz'

class Citation:
    @property
    def citation(self):
        return ("For citation, please use the following reference: "
                "https://github.com/chrisharhaw/timescape_astropy/tree/main' timescape_astropy (2024)")

citation_instance = Citation()
citation = citation_instance.citation