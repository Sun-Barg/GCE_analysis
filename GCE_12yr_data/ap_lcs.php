<!DOCTYPE html>
<html lang="en" class="usa">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>LAT 4FGL-DR2 Catalog Aperture Photometry Lightcurves</title>



<!-- USWDS CSS & JS -->
<link rel="stylesheet" href="/scripts/hds_pages/assets/uswds/css/styles.css" />
<script src="/scripts/hds_pages/assets/uswds/js/uswds-init.min.js"></script>

<link rel="stylesheet" href="/inc/css/about_fermi_hds.css">
<link rel="stylesheet" href="/inc/css/sub_template.css">
<link rel="stylesheet" href="/inc/css/side-banner.css" />

<style>
	/* Tidy list spacing inside callout */
	.callout ol {
	margin: 0.5rem 0 0 1.25rem;
	padding-left: 0;
	}
</style>

<!-- Google Analytics -->
<script type="text/javascript" language="javascript" id="_fed_an_ua_tag"
        src="https://dap.digitalgov.gov/Universal-Federated-Analytics-Min.js?agency=NASA">
</script>


</head>

<body>

  <!------------ shared header ------------>
<!-- Skip nav and header navigation -->
<a class="usa-skipnav" href="#main-content">Skip to main content</a>
<div class="usa-overlay"></div>
<header class="usa-header usa-header--extended border-top-1 border-secondary-dark bg-ink">
    <div class="usa-navbar">
        <div class="usa-logo">
            <div class="grid-container padding-0">
                <div class="grid-row">
                    <div class="grid-col-auto display-none desktop:display-block">
                        <a class="" href="https://www.nasa.gov">
                            <img src="/inc/img/nasa-logo.svg" class="maxw-10" alt="NASA Logo, National Aeronautics and Space Administration">
                        </a>
                    </div>
                    <div class="grid-col-fill">
                        <h6 class="margin-0 font-sans-2xs text-normal display-none desktop:display-block">
                            <a href="https://www.nasa.gov/"><span class="text-base-lightest">NASA | </span></a>
                            <a href="https://www.nasa.gov/goddard/"><span class="text-base-lightest">GSFC | </span></a>
                            <a href="https://science.gsfc.nasa.gov/"><span class="text-base-lightest">Sciences and Exploration</span></a>
                        </h6>
                        <h6 class="usa-logo__text font-sans-xl"><a href="/"><span
class="text-base-lightest">Fermi</span></a></h6>
                        <h6 class="margin-0 display-none desktop:display-block font-sans-2xs text-normal">
                            <a href="/"><span class="text-base-lightest">Gamma-ray Space Telescope</span></a>
                        </h6>
                    </div>
                </div>
            </div>
        </div>
        <button type="button" class="usa-menu-btn">Menu</button>
    </div>
    <nav aria-label="Primary navigation" class="usa-nav">
        <div class="usa-nav__inner">
            <button type="button" class="usa-nav__close">
                <img src="/inc/assets/uswds/img/usa-icons/close.svg" role="img" alt="Close" />
            </button>
            <ul class="usa-nav__primary usa-accordion">
                <li class="usa-nav__primary-item">
                    <button type="button" id="heasarc-nav" class="usa-accordion__button usa-nav__link" aria-expanded="false"
                        aria-controls="extended-nav-section-one">
                        <span>About </span>
                    </button>
                    <ul id="extended-nav-section-one" class="usa-nav__submenu">
                        <li class="usa-nav__submenu-item heasarc-nav-child">
                            <a href="/overview.html"><span>Overview</span></a>
                        </li>
                        <li class="usa-nav__submenu-item heasarc-nav-child">
                            <a href="https://imagine.gsfc.nasa.gov/observatories/learning/fermi/"><span>Educators</span></a>
                        </li>
                        <li class="usa-nav__submenu-item heasarc-nav-child">
                            <a href="https://science.nasa.gov/mission/fermi/"><span>Public</span></a>
                        </li>
                        <li class="usa-nav__submenu-item heasarc-nav-child">
                            <a href="https://svs.gsfc.nasa.gov/gallery/fermi5/"><span>Multimedia</span></a>
                        </li>
                        <li class="usa-nav__submenu-item heasarc-nav-child">
                            <a href="https://science.nasa.gov/missions/fermi/explore-the-universe-with-the-first-e-book-from-nasas-fermi/"><span>Science E-Book</span></a>
                        </li>
                        <li class="usa-nav__submenu-item heasarc-nav-child">
                            <a href="/science/"><span>Research Highlights</span></a>
                        </li>
                        <li class="usa-nav__submenu-item heasarc-nav-child">
                            <a href="/cgi-bin/bibliography_fermi"><span>Publications</span></a>
                        </li>
                        <li class="usa-nav__submenu-item heasarc-nav-child">
                            <a href="/contacts.html"><span>Contacts</span></a>
                        </li>
                    </ul>
                </li>
                <li class="usa-nav__primary-item">
                    <button type="button" id="observatories-nav" class="usa-accordion__button usa-nav__link" aria-expanded="false"
                        aria-controls="extended-nav-section-two">
                        <span>Science Support</span>
                    </button>
                    <ul id="extended-nav-section-two" class="usa-nav__submenu">
                        <li class="usa-nav__submenu-item observatories-nav-child text-bold border-bottom-1 border-primary">
                            <a href="/ssc/"><span>FSSC</span></a>
                        </li>
                        <li class="usa-nav__submenu-item observatories-nav-child">
                            <a href="/ssc/proposals/"><span>Proposers</span></a>
                        </li>
                        <li class="usa-nav__submenu-item observatories-nav-child">
                            <a href="/ssc/data/analysis/"><span>Tools & Software</span></a>
                        </li>
                        <li class="usa-nav__submenu-item observatories-nav-child">
                            <a href="/ssc/data/"><span>Data & Products</span></a>
                        </li>
                        <li class="usa-nav__submenu-item observatories-nav-child">
                            <a href="/new_to_fermi.html"><span>New Users</span></a>
                        </li>
                        <li class="usa-nav__submenu-item observatories-nav-child">
                            <a href="/ssc/observations/multi/programs.html"><span>Multiwavelength Resources</span></a>
                        </li>
                        <li class="usa-nav__submenu-item observatories-nav-child">
                            <a href="/ssc/multimessenger.html"><span>Time-domain Resources</span></a>
                        </li>
                        <li class="usa-nav__submenu-item observatories-nav-child">
                            <a href="/ssc/help/about.html"><span>FSSC People</span></a>
                        </li>
                        <li class="usa-nav__submenu-item observatories-nav-child">
                            <a href="/ssc/help/"><span>Helpdesk</span></a>
                        </li>
                    </ul>
                </li>
                <li class="usa-nav__primary-item">
                    <button type="button" id="archive-nav" class="usa-accordion__button usa-nav__link" aria-expanded="false"
                        aria-controls="extended-nav-section-three">
                        <span>Observatory</span>
                    </button>
                    <ul id="extended-nav-section-three" class="usa-nav__submenu">
                        <li class="usa-nav__submenu-item archive-nav-child">
                            <a href="/ssc/observations/"><span>Overview</span></a>
                        </li>
                        <li class="usa-nav__submenu-item archive-nav-child">
                            <a href="/science/instruments/lat.html"><span>LAT</span></a>
                        </li>
                        <li class="usa-nav__submenu-item archive-nav-child">
                            <a href="/science/instruments/gbm.html"><span>GBM</span></a>
                        </li>
                        <li class="usa-nav__submenu-item archive-nav-child">
                            <a href="/ssc/observations/timeline/posting/"><span>Timeline</span></a>
                        </li>
                        <li class="usa-nav__submenu-item archive-nav-child">
                            <a href="/ssc/observations/types/allsky/"><span>Survey Modes</span></a>
                        </li>
                        <li class="usa-nav__submenu-item archive-nav-child">
                            <a href="/ssc/observations/types/too/"><span>Targets of Opportunity</span></a>
                        </li>
                    </ul>
                </li>

                <li class="usa-nav__primary-item">
                    <button type="button" id="calibration-nav" class="usa-accordion__button usa-nav__link" aria-expanded="false"
                        aria-controls="extended-nav-section-four">
                        <span>News</span>
                    </button>
                    <ul id="extended-nav-section-four" class="usa-nav__submenu">
                        <li class="usa-nav__submenu-item calibration-nav-child">
                            <a href="https://science.nasa.gov/mission/fermi/stories/"><span>Press</span></a>

                        </li>
                        <li class="usa-nav__submenu-item calibration-nav-child">
                            <a href="/ssc/news/"><span>Announcements</span></a>
                        </li>
						<li class="usa-nav__submenu-item calibration-nav-child">
                            <a href="/ssc/mailing_lists.html"><span>News and Mailing Lists</span></a>
                        </li>
                    </ul>
                </li>

                <li class="usa-nav__primary-item">
                    <button type="button" id="software-nav" class="usa-accordion__button usa-nav__link" aria-expanded="false"
                        aria-controls="extended-nav-section-five">
                        <span>Community</span>
                    </button>
                    <ul id="extended-nav-section-five" class="usa-nav__submenu">
                        <li class="usa-nav__submenu-item software-nav-child">
                            <a href="/science/mtgs/"><span>Meetings and Workshops</span></a>
                        </li>
                        <li class="usa-nav__submenu-item software-nav-child">
                            <a href="/science/mtgs/#symposia-panel"><span>Symposia</span></a>
                        </li>
                        <li class="usa-nav__submenu-item software-nav-child">
                            <a href="/science/mtgs/summerschool/"><span>Summer School</span></a>
                        </li>
                        <li class="usa-nav__submenu-item software-nav-child">
                            <a href="/ssc/users_group/"><span>Users Group</span></a>
                        </li>
                        <!--li class="usa-nav__submenu-item software-nav-child">
                            <a href="/ssc/observations/multi/programs.html"><span>Multiwavelength Coordination</span></a>
                        </li-->
                    </ul>
                </li>


            </ul>
            <div class="usa-nav__secondary">
                <ul class="usa-nav__secondary-links">
                    <li class="usa-nav__secondary-item"><a href="/"><span>Fermi</span></a></li>
                    <li class="usa-nav__secondary-item"><a href="/ssc/"><span>FSSC</span></a></li>
                    <li class="usa-nav__secondary-item"><a href="http://heasarc.gsfc.nasa.gov/"><span>HEASARC</span></a></li>
                </ul>
                <section aria-label="Search component">
                    <form class="usa-search usa-search--small" role="search" action="https://heasarc.gsfc.nasa.gov/cgi-bin/search/search.pl" method="post" id="HEASARCsearch" name="HEASARCsearch">
                        <label class="usa-sr-only" for="search">Search</label>
                        <input class="usa-input radius-left-md" id="search" type="search" name="tquery" />
                        <button class="usa-button usa-button--secondary usa-button--hover" type="submit">
                            <img src="/inc/assets/uswds/img/usa-icons-bg/search--white.svg" class="usa-search__submit-icon"
                                alt="Search" />
                        </button>
                    </form>
                </section>
            </div>
        </div>
    </nav>
</header>
<!--#include virtual="downtime.html" -->

  <main class="usa-section" id="main-content">

<!-- Start Section Wrapper -->
<div id="sec-wrapper">

<h1>LAT 4FGL-DR2 Catalog Aperture Photometry Lightcurves</h1>

<p>
We provide here aperture photometry light curves of all sources in the Fermi Large Area Telescope 10-year Source Catalog (4FGL-DR2) with 30 day time resolution. The light curves and plots will generally be updated weekly. We also plot power spectra of the light curves for periods between 65 days and the length of the light curve, with a frequency oversampling of a factor of 5.
</p>

<div class="callout">
	<p>
	Some caveats:
	</p>
	<ul>
		<li>These light curves are intended to provide a quick way to inspect gross features in the long-term light curves. They are unlikely to be directly useful for detailed scientific analysis.</li>
		<li>They are not background subtracted.</li>
		<li>The apertures will contain photons from nearby sources.</li>
	</ul>
</div>
<p>
A thread describing the steps involved in aperture photometry of LAT data can be found <a href="/ssc/data/analysis/scitools/aperture_photometry.html">here</a>. A script to perform aperture photometry (aperture.pl) is available <a href="/ssc/data/analysis/user/">here</a>.
</p>
<div class="callout">
	<p>
	The columns in the data files are:
	</p>
	<ol>
		<li>Time of center of bin (MJD)</li>
		<li>Count rate in the aperture in photons cm^-2 s^-1</li>
		<li>Count rate error based on the observed number of photons. This is used in the plots</li>
		<li>Half-width of the time bin in days.</li>
		<li>Count rate error based on the relative exposure of the time bin.</li>
		<li>Exposure in cm^2 s.</li>
	</ol>
</div>
<div class="callout">
	<p>
	The aperture photometry used:
	</p>
	<ul>
		<li>1 degree radius apertures</li>
		<li>100 MeV to 200 GeV</li>
		<li>Data are P8 Source class (evclass=128), front+back (evtype=3)</li>
		<li>A zenith limit of 105 degrees.</li>
		<li>A rock limit of 90 degrees. (i.e. no filtering on this.)</li>
		<li>A bore-sight limit of 180 degrees. (i.e. no filtering on this.)</li>
		<li>Times when a source was closer than 5 degrees from the Sun are excluded.</li>
	</ul>
</div>
<p>For caveats on period detection in LAT light curves please see <a href="/ssc/data/analysis/LAT_caveats_temporal.html">here</a>.</p>

<p class="center">
<b><a href="/ssc/data/access/lat/10yr_catalog/ap_lcs_flares.html">Flaring Sources in the Aperture Photometry Lightcurves</a></b>
</p>

<hr>

<p style="text-align: center;"><b>RA Range:</b><br><a href="ap_lcs.php?ra=00-01">00-01</a> | <a href="ap_lcs.php?ra=02-03">02-03</a> | <a href="ap_lcs.php?ra=04-05">04-05</a> | <a href="ap_lcs.php?ra=06-07">06-07</a> | <a href="ap_lcs.php?ra=08-09">08-09</a> | <a href="ap_lcs.php?ra=10-11">10-11</a> | <a href="ap_lcs.php?ra=12-13">12-13</a> | <a href="ap_lcs.php?ra=14-15">14-15</a> | <a href="ap_lcs.php?ra=16-17">16-17</a> | <a href="ap_lcs.php?ra=18-19">18-19</a> | <a href="ap_lcs.php?ra=20-21">20-21</a> | <a href="ap_lcs.php?ra=22-23">22-23</a></p>
<div style="display: flex; flex-wrap: wrap; justify-content: center;">
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0000.3-7355</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0000.3-7355.png"><img src="ap_lcs/lightcurve_4FGLJ0000.3-7355.png" width="200" height="155" alt="lightcurve_4FGLJ0000.3-7355.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0000.3-7355.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0000.5+0743</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0000.5p0743.png"><img src="ap_lcs/lightcurve_4FGLJ0000.5p0743.png" width="200" height="155" alt="lightcurve_4FGLJ0000.5p0743.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0000.5p0743.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0001.2-0747</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0001.2-0747.png"><img src="ap_lcs/lightcurve_4FGLJ0001.2-0747.png" width="200" height="155" alt="lightcurve_4FGLJ0001.2-0747.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0001.2-0747.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0001.2+4741</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0001.2p4741.png"><img src="ap_lcs/lightcurve_4FGLJ0001.2p4741.png" width="200" height="155" alt="lightcurve_4FGLJ0001.2p4741.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0001.2p4741.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0001.5+2113</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0001.5p2113.png"><img src="ap_lcs/lightcurve_4FGLJ0001.5p2113.png" width="200" height="155" alt="lightcurve_4FGLJ0001.5p2113.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0001.5p2113.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0001.6-4156</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0001.6-4156.png"><img src="ap_lcs/lightcurve_4FGLJ0001.6-4156.png" width="200" height="155" alt="lightcurve_4FGLJ0001.6-4156.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0001.6-4156.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0002.1-6728</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0002.1-6728.png"><img src="ap_lcs/lightcurve_4FGLJ0002.1-6728.png" width="200" height="155" alt="lightcurve_4FGLJ0002.1-6728.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0002.1-6728.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0002.1+6721c</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0002.1p6721c.png"><img src="ap_lcs/lightcurve_4FGLJ0002.1p6721c.png" width="200" height="155" alt="lightcurve_4FGLJ0002.1p6721c.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0002.1p6721c.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0002.3-0815</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0002.3-0815.png"><img src="ap_lcs/lightcurve_4FGLJ0002.3-0815.png" width="200" height="155" alt="lightcurve_4FGLJ0002.3-0815.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0002.3-0815.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0002.4-5156</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0002.4-5156.png"><img src="ap_lcs/lightcurve_4FGLJ0002.4-5156.png" width="200" height="155" alt="lightcurve_4FGLJ0002.4-5156.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0002.4-5156.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0002.7+7220</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0002.7p7220.png"><img src="ap_lcs/lightcurve_4FGLJ0002.7p7220.png" width="200" height="155" alt="lightcurve_4FGLJ0002.7p7220.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0002.7p7220.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0002.8+6217</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0002.8p6217.png"><img src="ap_lcs/lightcurve_4FGLJ0002.8p6217.png" width="200" height="155" alt="lightcurve_4FGLJ0002.8p6217.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0002.8p6217.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0003.1-5248</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0003.1-5248.png"><img src="ap_lcs/lightcurve_4FGLJ0003.1-5248.png" width="200" height="155" alt="lightcurve_4FGLJ0003.1-5248.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0003.1-5248.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0003.2+2207</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0003.2p2207.png"><img src="ap_lcs/lightcurve_4FGLJ0003.2p2207.png" width="200" height="155" alt="lightcurve_4FGLJ0003.2p2207.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0003.2p2207.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0003.3-1928</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0003.3-1928.png"><img src="ap_lcs/lightcurve_4FGLJ0003.3-1928.png" width="200" height="155" alt="lightcurve_4FGLJ0003.3-1928.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0003.3-1928.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0003.3-5905</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0003.3-5905.png"><img src="ap_lcs/lightcurve_4FGLJ0003.3-5905.png" width="200" height="155" alt="lightcurve_4FGLJ0003.3-5905.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0003.3-5905.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0003.3+2511</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0003.3p2511.png"><img src="ap_lcs/lightcurve_4FGLJ0003.3p2511.png" width="200" height="155" alt="lightcurve_4FGLJ0003.3p2511.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0003.3p2511.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0003.6+3059</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0003.6p3059.png"><img src="ap_lcs/lightcurve_4FGLJ0003.6p3059.png" width="200" height="155" alt="lightcurve_4FGLJ0003.6p3059.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0003.6p3059.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0003.9-1149</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0003.9-1149.png"><img src="ap_lcs/lightcurve_4FGLJ0003.9-1149.png" width="200" height="155" alt="lightcurve_4FGLJ0003.9-1149.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0003.9-1149.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0004.0+0840</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0004.0p0840.png"><img src="ap_lcs/lightcurve_4FGLJ0004.0p0840.png" width="200" height="155" alt="lightcurve_4FGLJ0004.0p0840.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0004.0p0840.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0004.0+5715</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0004.0p5715.png"><img src="ap_lcs/lightcurve_4FGLJ0004.0p5715.png" width="200" height="155" alt="lightcurve_4FGLJ0004.0p5715.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0004.0p5715.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0004.3+4614</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0004.3p4614.png"><img src="ap_lcs/lightcurve_4FGLJ0004.3p4614.png" width="200" height="155" alt="lightcurve_4FGLJ0004.3p4614.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0004.3p4614.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0004.4-4001</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0004.4-4001.png"><img src="ap_lcs/lightcurve_4FGLJ0004.4-4001.png" width="200" height="155" alt="lightcurve_4FGLJ0004.4-4001.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0004.4-4001.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0004.4-4737</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0004.4-4737.png"><img src="ap_lcs/lightcurve_4FGLJ0004.4-4737.png" width="200" height="155" alt="lightcurve_4FGLJ0004.4-4737.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0004.4-4737.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0005.6+6746c</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0005.6p6746c.png"><img src="ap_lcs/lightcurve_4FGLJ0005.6p6746c.png" width="200" height="155" alt="lightcurve_4FGLJ0005.6p6746c.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0005.6p6746c.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0005.9+3824</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0005.9p3824.png"><img src="ap_lcs/lightcurve_4FGLJ0005.9p3824.png" width="200" height="155" alt="lightcurve_4FGLJ0005.9p3824.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0005.9p3824.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0006.3-0620</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0006.3-0620.png"><img src="ap_lcs/lightcurve_4FGLJ0006.3-0620.png" width="200" height="155" alt="lightcurve_4FGLJ0006.3-0620.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0006.3-0620.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0006.4+0135</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0006.4p0135.png"><img src="ap_lcs/lightcurve_4FGLJ0006.4p0135.png" width="200" height="155" alt="lightcurve_4FGLJ0006.4p0135.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0006.4p0135.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0006.6+4618</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0006.6p4618.png"><img src="ap_lcs/lightcurve_4FGLJ0006.6p4618.png" width="200" height="155" alt="lightcurve_4FGLJ0006.6p4618.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0006.6p4618.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0007.0+7303</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0007.0p7303.png"><img src="ap_lcs/lightcurve_4FGLJ0007.0p7303.png" width="200" height="155" alt="lightcurve_4FGLJ0007.0p7303.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0007.0p7303.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0007.7+4008</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0007.7p4008.png"><img src="ap_lcs/lightcurve_4FGLJ0007.7p4008.png" width="200" height="155" alt="lightcurve_4FGLJ0007.7p4008.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0007.7p4008.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0008.0-3937</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0008.0-3937.png"><img src="ap_lcs/lightcurve_4FGLJ0008.0-3937.png" width="200" height="155" alt="lightcurve_4FGLJ0008.0-3937.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0008.0-3937.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0008.0+4711</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0008.0p4711.png"><img src="ap_lcs/lightcurve_4FGLJ0008.0p4711.png" width="200" height="155" alt="lightcurve_4FGLJ0008.0p4711.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0008.0p4711.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0008.4-2339</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0008.4-2339.png"><img src="ap_lcs/lightcurve_4FGLJ0008.4-2339.png" width="200" height="155" alt="lightcurve_4FGLJ0008.4-2339.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0008.4-2339.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0008.4+1455</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0008.4p1455.png"><img src="ap_lcs/lightcurve_4FGLJ0008.4p1455.png" width="200" height="155" alt="lightcurve_4FGLJ0008.4p1455.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0008.4p1455.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0008.4+6926</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0008.4p6926.png"><img src="ap_lcs/lightcurve_4FGLJ0008.4p6926.png" width="200" height="155" alt="lightcurve_4FGLJ0008.4p6926.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0008.4p6926.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0008.9+2509</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0008.9p2509.png"><img src="ap_lcs/lightcurve_4FGLJ0008.9p2509.png" width="200" height="155" alt="lightcurve_4FGLJ0008.9p2509.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0008.9p2509.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0009.1-5012</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0009.1-5012.png"><img src="ap_lcs/lightcurve_4FGLJ0009.1-5012.png" width="200" height="155" alt="lightcurve_4FGLJ0009.1-5012.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0009.1-5012.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0009.1+0628</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0009.1p0628.png"><img src="ap_lcs/lightcurve_4FGLJ0009.1p0628.png" width="200" height="155" alt="lightcurve_4FGLJ0009.1p0628.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0009.1p0628.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0009.2+1745</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0009.2p1745.png"><img src="ap_lcs/lightcurve_4FGLJ0009.2p1745.png" width="200" height="155" alt="lightcurve_4FGLJ0009.2p1745.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0009.2p1745.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0009.2+6847</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0009.2p6847.png"><img src="ap_lcs/lightcurve_4FGLJ0009.2p6847.png" width="200" height="155" alt="lightcurve_4FGLJ0009.2p6847.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0009.2p6847.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0009.3+5030</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0009.3p5030.png"><img src="ap_lcs/lightcurve_4FGLJ0009.3p5030.png" width="200" height="155" alt="lightcurve_4FGLJ0009.3p5030.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0009.3p5030.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0009.7-1418</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0009.7-1418.png"><img src="ap_lcs/lightcurve_4FGLJ0009.7-1418.png" width="200" height="155" alt="lightcurve_4FGLJ0009.7-1418.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0009.7-1418.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0009.7-3217</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0009.7-3217.png"><img src="ap_lcs/lightcurve_4FGLJ0009.7-3217.png" width="200" height="155" alt="lightcurve_4FGLJ0009.7-3217.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0009.7-3217.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0009.8-4317</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0009.8-4317.png"><img src="ap_lcs/lightcurve_4FGLJ0009.8-4317.png" width="200" height="155" alt="lightcurve_4FGLJ0009.8-4317.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0009.8-4317.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0009.8+1340</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0009.8p1340.png"><img src="ap_lcs/lightcurve_4FGLJ0009.8p1340.png" width="200" height="155" alt="lightcurve_4FGLJ0009.8p1340.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0009.8p1340.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0010.2-2431</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0010.2-2431.png"><img src="ap_lcs/lightcurve_4FGLJ0010.2-2431.png" width="200" height="155" alt="lightcurve_4FGLJ0010.2-2431.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0010.2-2431.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0010.6-3025</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0010.6-3025.png"><img src="ap_lcs/lightcurve_4FGLJ0010.6-3025.png" width="200" height="155" alt="lightcurve_4FGLJ0010.6-3025.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0010.6-3025.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0010.6+2043</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0010.6p2043.png"><img src="ap_lcs/lightcurve_4FGLJ0010.6p2043.png" width="200" height="155" alt="lightcurve_4FGLJ0010.6p2043.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0010.6p2043.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0010.8-2154</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0010.8-2154.png"><img src="ap_lcs/lightcurve_4FGLJ0010.8-2154.png" width="200" height="155" alt="lightcurve_4FGLJ0010.8-2154.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0010.8-2154.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0011.4-4110</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0011.4-4110.png"><img src="ap_lcs/lightcurve_4FGLJ0011.4-4110.png" width="200" height="155" alt="lightcurve_4FGLJ0011.4-4110.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0011.4-4110.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0011.4+0057</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0011.4p0057.png"><img src="ap_lcs/lightcurve_4FGLJ0011.4p0057.png" width="200" height="155" alt="lightcurve_4FGLJ0011.4p0057.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0011.4p0057.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0011.8-3142</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0011.8-3142.png"><img src="ap_lcs/lightcurve_4FGLJ0011.8-3142.png" width="200" height="155" alt="lightcurve_4FGLJ0011.8-3142.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0011.8-3142.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0012.0+7043</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0012.0p7043.png"><img src="ap_lcs/lightcurve_4FGLJ0012.0p7043.png" width="200" height="155" alt="lightcurve_4FGLJ0012.0p7043.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0012.0p7043.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0013.1-3955</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0013.1-3955.png"><img src="ap_lcs/lightcurve_4FGLJ0013.1-3955.png" width="200" height="155" alt="lightcurve_4FGLJ0013.1-3955.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0013.1-3955.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0013.4+0950</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0013.4p0950.png"><img src="ap_lcs/lightcurve_4FGLJ0013.4p0950.png" width="200" height="155" alt="lightcurve_4FGLJ0013.4p0950.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0013.4p0950.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0013.6-0424</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0013.6-0424.png"><img src="ap_lcs/lightcurve_4FGLJ0013.6-0424.png" width="200" height="155" alt="lightcurve_4FGLJ0013.6-0424.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0013.6-0424.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0013.6+4051</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0013.6p4051.png"><img src="ap_lcs/lightcurve_4FGLJ0013.6p4051.png" width="200" height="155" alt="lightcurve_4FGLJ0013.6p4051.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0013.6p4051.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0013.9-1854</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0013.9-1854.png"><img src="ap_lcs/lightcurve_4FGLJ0013.9-1854.png" width="200" height="155" alt="lightcurve_4FGLJ0013.9-1854.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0013.9-1854.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0014.1-5022</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0014.1-5022.png"><img src="ap_lcs/lightcurve_4FGLJ0014.1-5022.png" width="200" height="155" alt="lightcurve_4FGLJ0014.1-5022.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0014.1-5022.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0014.1+1910</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0014.1p1910.png"><img src="ap_lcs/lightcurve_4FGLJ0014.1p1910.png" width="200" height="155" alt="lightcurve_4FGLJ0014.1p1910.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0014.1p1910.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0014.2+0854</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0014.2p0854.png"><img src="ap_lcs/lightcurve_4FGLJ0014.2p0854.png" width="200" height="155" alt="lightcurve_4FGLJ0014.2p0854.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0014.2p0854.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0014.3-0500</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0014.3-0500.png"><img src="ap_lcs/lightcurve_4FGLJ0014.3-0500.png" width="200" height="155" alt="lightcurve_4FGLJ0014.3-0500.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0014.3-0500.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0014.7+5801</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0014.7p5801.png"><img src="ap_lcs/lightcurve_4FGLJ0014.7p5801.png" width="200" height="155" alt="lightcurve_4FGLJ0014.7p5801.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0014.7p5801.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0014.8+6118</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0014.8p6118.png"><img src="ap_lcs/lightcurve_4FGLJ0014.8p6118.png" width="200" height="155" alt="lightcurve_4FGLJ0014.8p6118.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0014.8p6118.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0014.9+3212</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0014.9p3212.png"><img src="ap_lcs/lightcurve_4FGLJ0014.9p3212.png" width="200" height="155" alt="lightcurve_4FGLJ0014.9p3212.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0014.9p3212.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0015.2+3537</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0015.2p3537.png"><img src="ap_lcs/lightcurve_4FGLJ0015.2p3537.png" width="200" height="155" alt="lightcurve_4FGLJ0015.2p3537.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0015.2p3537.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0015.6+5551</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0015.6p5551.png"><img src="ap_lcs/lightcurve_4FGLJ0015.6p5551.png" width="200" height="155" alt="lightcurve_4FGLJ0015.6p5551.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0015.6p5551.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0015.9+2440</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0015.9p2440.png"><img src="ap_lcs/lightcurve_4FGLJ0015.9p2440.png" width="200" height="155" alt="lightcurve_4FGLJ0015.9p2440.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0015.9p2440.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0016.2-0016</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0016.2-0016.png"><img src="ap_lcs/lightcurve_4FGLJ0016.2-0016.png" width="200" height="155" alt="lightcurve_4FGLJ0016.2-0016.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0016.2-0016.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0016.5+1702</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0016.5p1702.png"><img src="ap_lcs/lightcurve_4FGLJ0016.5p1702.png" width="200" height="155" alt="lightcurve_4FGLJ0016.5p1702.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0016.5p1702.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0017.0-0649</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0017.0-0649.png"><img src="ap_lcs/lightcurve_4FGLJ0017.0-0649.png" width="200" height="155" alt="lightcurve_4FGLJ0017.0-0649.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0017.0-0649.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0017.1-4605</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0017.1-4605.png"><img src="ap_lcs/lightcurve_4FGLJ0017.1-4605.png" width="200" height="155" alt="lightcurve_4FGLJ0017.1-4605.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0017.1-4605.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0017.5-0514</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0017.5-0514.png"><img src="ap_lcs/lightcurve_4FGLJ0017.5-0514.png" width="200" height="155" alt="lightcurve_4FGLJ0017.5-0514.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0017.5-0514.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0017.8+1455</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0017.8p1455.png"><img src="ap_lcs/lightcurve_4FGLJ0017.8p1455.png" width="200" height="155" alt="lightcurve_4FGLJ0017.8p1455.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0017.8p1455.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0018.4+2946</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0018.4p2946.png"><img src="ap_lcs/lightcurve_4FGLJ0018.4p2946.png" width="200" height="155" alt="lightcurve_4FGLJ0018.4p2946.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0018.4p2946.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0019.2-5640</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0019.2-5640.png"><img src="ap_lcs/lightcurve_4FGLJ0019.2-5640.png" width="200" height="155" alt="lightcurve_4FGLJ0019.2-5640.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0019.2-5640.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0019.3-8152</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0019.3-8152.png"><img src="ap_lcs/lightcurve_4FGLJ0019.3-8152.png" width="200" height="155" alt="lightcurve_4FGLJ0019.3-8152.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0019.3-8152.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0019.6+2022</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0019.6p2022.png"><img src="ap_lcs/lightcurve_4FGLJ0019.6p2022.png" width="200" height="155" alt="lightcurve_4FGLJ0019.6p2022.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0019.6p2022.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0019.6+7327</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0019.6p7327.png"><img src="ap_lcs/lightcurve_4FGLJ0019.6p7327.png" width="200" height="155" alt="lightcurve_4FGLJ0019.6p7327.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0019.6p7327.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0020.3+6919</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0020.3p6919.png"><img src="ap_lcs/lightcurve_4FGLJ0020.3p6919.png" width="200" height="155" alt="lightcurve_4FGLJ0020.3p6919.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0020.3p6919.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0021.0+0322</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0021.0p0322.png"><img src="ap_lcs/lightcurve_4FGLJ0021.0p0322.png" width="200" height="155" alt="lightcurve_4FGLJ0021.0p0322.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0021.0p0322.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0021.5-2552</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0021.5-2552.png"><img src="ap_lcs/lightcurve_4FGLJ0021.5-2552.png" width="200" height="155" alt="lightcurve_4FGLJ0021.5-2552.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0021.5-2552.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0021.6-0855</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0021.6-0855.png"><img src="ap_lcs/lightcurve_4FGLJ0021.6-0855.png" width="200" height="155" alt="lightcurve_4FGLJ0021.6-0855.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0021.6-0855.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0021.9-5140</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0021.9-5140.png"><img src="ap_lcs/lightcurve_4FGLJ0021.9-5140.png" width="200" height="155" alt="lightcurve_4FGLJ0021.9-5140.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0021.9-5140.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0022.0-5921</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0022.0-5921.png"><img src="ap_lcs/lightcurve_4FGLJ0022.0-5921.png" width="200" height="155" alt="lightcurve_4FGLJ0022.0-5921.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0022.0-5921.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0022.0+0006</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0022.0p0006.png"><img src="ap_lcs/lightcurve_4FGLJ0022.0p0006.png" width="200" height="155" alt="lightcurve_4FGLJ0022.0p0006.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0022.0p0006.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0022.1-1854</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0022.1-1854.png"><img src="ap_lcs/lightcurve_4FGLJ0022.1-1854.png" width="200" height="155" alt="lightcurve_4FGLJ0022.1-1854.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0022.1-1854.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0022.5+0608</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0022.5p0608.png"><img src="ap_lcs/lightcurve_4FGLJ0022.5p0608.png" width="200" height="155" alt="lightcurve_4FGLJ0022.5p0608.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0022.5p0608.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0023.4+0920</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0023.4p0920.png"><img src="ap_lcs/lightcurve_4FGLJ0023.4p0920.png" width="200" height="155" alt="lightcurve_4FGLJ0023.4p0920.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0023.4p0920.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0023.6-4209</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0023.6-4209.png"><img src="ap_lcs/lightcurve_4FGLJ0023.6-4209.png" width="200" height="155" alt="lightcurve_4FGLJ0023.6-4209.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0023.6-4209.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0023.7-6820</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0023.7-6820.png"><img src="ap_lcs/lightcurve_4FGLJ0023.7-6820.png" width="200" height="155" alt="lightcurve_4FGLJ0023.7-6820.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0023.7-6820.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0023.7+4457</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0023.7p4457.png"><img src="ap_lcs/lightcurve_4FGLJ0023.7p4457.png" width="200" height="155" alt="lightcurve_4FGLJ0023.7p4457.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0023.7p4457.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0023.9+1603</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0023.9p1603.png"><img src="ap_lcs/lightcurve_4FGLJ0023.9p1603.png" width="200" height="155" alt="lightcurve_4FGLJ0023.9p1603.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0023.9p1603.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0024.0-7204</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0024.0-7204.png"><img src="ap_lcs/lightcurve_4FGLJ0024.0-7204.png" width="200" height="155" alt="lightcurve_4FGLJ0024.0-7204.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0024.0-7204.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0024.1+2402</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0024.1p2402.png"><img src="ap_lcs/lightcurve_4FGLJ0024.1p2402.png" width="200" height="155" alt="lightcurve_4FGLJ0024.1p2402.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0024.1p2402.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0024.4+4647</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0024.4p4647.png"><img src="ap_lcs/lightcurve_4FGLJ0024.4p4647.png" width="200" height="155" alt="lightcurve_4FGLJ0024.4p4647.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0024.4p4647.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0024.7+0349</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0024.7p0349.png"><img src="ap_lcs/lightcurve_4FGLJ0024.7p0349.png" width="200" height="155" alt="lightcurve_4FGLJ0024.7p0349.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0024.7p0349.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0025.2-2231</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0025.2-2231.png"><img src="ap_lcs/lightcurve_4FGLJ0025.2-2231.png" width="200" height="155" alt="lightcurve_4FGLJ0025.2-2231.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0025.2-2231.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0025.3+6408</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0025.3p6408.png"><img src="ap_lcs/lightcurve_4FGLJ0025.3p6408.png" width="200" height="155" alt="lightcurve_4FGLJ0025.3p6408.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0025.3p6408.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0025.4-4838</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0025.4-4838.png"><img src="ap_lcs/lightcurve_4FGLJ0025.4-4838.png" width="200" height="155" alt="lightcurve_4FGLJ0025.4-4838.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0025.4-4838.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0025.5-5936</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0025.5-5936.png"><img src="ap_lcs/lightcurve_4FGLJ0025.5-5936.png" width="200" height="155" alt="lightcurve_4FGLJ0025.5-5936.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0025.5-5936.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0025.7-4801</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0025.7-4801.png"><img src="ap_lcs/lightcurve_4FGLJ0025.7-4801.png" width="200" height="155" alt="lightcurve_4FGLJ0025.7-4801.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0025.7-4801.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0026.1-0732</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0026.1-0732.png"><img src="ap_lcs/lightcurve_4FGLJ0026.1-0732.png" width="200" height="155" alt="lightcurve_4FGLJ0026.1-0732.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0026.1-0732.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0026.6-4600</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0026.6-4600.png"><img src="ap_lcs/lightcurve_4FGLJ0026.6-4600.png" width="200" height="155" alt="lightcurve_4FGLJ0026.6-4600.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0026.6-4600.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0028.1+7505</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0028.1p7505.png"><img src="ap_lcs/lightcurve_4FGLJ0028.1p7505.png" width="200" height="155" alt="lightcurve_4FGLJ0028.1p7505.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0028.1p7505.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0028.4+2001</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0028.4p2001.png"><img src="ap_lcs/lightcurve_4FGLJ0028.4p2001.png" width="200" height="155" alt="lightcurve_4FGLJ0028.4p2001.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0028.4p2001.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0028.8-0112</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0028.8-0112.png"><img src="ap_lcs/lightcurve_4FGLJ0028.8-0112.png" width="200" height="155" alt="lightcurve_4FGLJ0028.8-0112.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0028.8-0112.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0028.9+3553</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0028.9p3553.png"><img src="ap_lcs/lightcurve_4FGLJ0028.9p3553.png" width="200" height="155" alt="lightcurve_4FGLJ0028.9p3553.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0028.9p3553.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0029.0-7044</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0029.0-7044.png"><img src="ap_lcs/lightcurve_4FGLJ0029.0-7044.png" width="200" height="155" alt="lightcurve_4FGLJ0029.0-7044.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0029.0-7044.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0029.4+2051</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0029.4p2051.png"><img src="ap_lcs/lightcurve_4FGLJ0029.4p2051.png" width="200" height="155" alt="lightcurve_4FGLJ0029.4p2051.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0029.4p2051.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0030.2-1647</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0030.2-1647.png"><img src="ap_lcs/lightcurve_4FGLJ0030.2-1647.png" width="200" height="155" alt="lightcurve_4FGLJ0030.2-1647.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0030.2-1647.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0030.3-4224</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0030.3-4224.png"><img src="ap_lcs/lightcurve_4FGLJ0030.3-4224.png" width="200" height="155" alt="lightcurve_4FGLJ0030.3-4224.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0030.3-4224.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0030.4+0451</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0030.4p0451.png"><img src="ap_lcs/lightcurve_4FGLJ0030.4p0451.png" width="200" height="155" alt="lightcurve_4FGLJ0030.4p0451.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0030.4p0451.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0030.6-0212</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0030.6-0212.png"><img src="ap_lcs/lightcurve_4FGLJ0030.6-0212.png" width="200" height="155" alt="lightcurve_4FGLJ0030.6-0212.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0030.6-0212.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0030.6+0539</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0030.6p0539.png"><img src="ap_lcs/lightcurve_4FGLJ0030.6p0539.png" width="200" height="155" alt="lightcurve_4FGLJ0030.6p0539.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0030.6p0539.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0030.9-3618</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0030.9-3618.png"><img src="ap_lcs/lightcurve_4FGLJ0030.9-3618.png" width="200" height="155" alt="lightcurve_4FGLJ0030.9-3618.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0030.9-3618.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0031.0-2327</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0031.0-2327.png"><img src="ap_lcs/lightcurve_4FGLJ0031.0-2327.png" width="200" height="155" alt="lightcurve_4FGLJ0031.0-2327.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0031.0-2327.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0031.3+0726</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0031.3p0726.png"><img src="ap_lcs/lightcurve_4FGLJ0031.3p0726.png" width="200" height="155" alt="lightcurve_4FGLJ0031.3p0726.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0031.3p0726.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0031.5-5648</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0031.5-5648.png"><img src="ap_lcs/lightcurve_4FGLJ0031.5-5648.png" width="200" height="155" alt="lightcurve_4FGLJ0031.5-5648.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0031.5-5648.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0032.3-5522</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0032.3-5522.png"><img src="ap_lcs/lightcurve_4FGLJ0032.3-5522.png" width="200" height="155" alt="lightcurve_4FGLJ0032.3-5522.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0032.3-5522.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0032.3-5539</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0032.3-5539.png"><img src="ap_lcs/lightcurve_4FGLJ0032.3-5539.png" width="200" height="155" alt="lightcurve_4FGLJ0032.3-5539.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0032.3-5539.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0032.4-2849</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0032.4-2849.png"><img src="ap_lcs/lightcurve_4FGLJ0032.4-2849.png" width="200" height="155" alt="lightcurve_4FGLJ0032.4-2849.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0032.4-2849.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0033.3-2040</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0033.3-2040.png"><img src="ap_lcs/lightcurve_4FGLJ0033.3-2040.png" width="200" height="155" alt="lightcurve_4FGLJ0033.3-2040.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0033.3-2040.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0033.5-1921</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0033.5-1921.png"><img src="ap_lcs/lightcurve_4FGLJ0033.5-1921.png" width="200" height="155" alt="lightcurve_4FGLJ0033.5-1921.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0033.5-1921.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0033.9+3858</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0033.9p3858.png"><img src="ap_lcs/lightcurve_4FGLJ0033.9p3858.png" width="200" height="155" alt="lightcurve_4FGLJ0033.9p3858.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0033.9p3858.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0034.0-4116</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0034.0-4116.png"><img src="ap_lcs/lightcurve_4FGLJ0034.0-4116.png" width="200" height="155" alt="lightcurve_4FGLJ0034.0-4116.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0034.0-4116.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0034.3-0534</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0034.3-0534.png"><img src="ap_lcs/lightcurve_4FGLJ0034.3-0534.png" width="200" height="155" alt="lightcurve_4FGLJ0034.3-0534.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0034.3-0534.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0034.6+6438</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0034.6p6438.png"><img src="ap_lcs/lightcurve_4FGLJ0034.6p6438.png" width="200" height="155" alt="lightcurve_4FGLJ0034.6p6438.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0034.6p6438.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0035.0-5728</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0035.0-5728.png"><img src="ap_lcs/lightcurve_4FGLJ0035.0-5728.png" width="200" height="155" alt="lightcurve_4FGLJ0035.0-5728.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0035.0-5728.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0035.2-1739</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0035.2-1739.png"><img src="ap_lcs/lightcurve_4FGLJ0035.2-1739.png" width="200" height="155" alt="lightcurve_4FGLJ0035.2-1739.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0035.2-1739.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0035.2+1514</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0035.2p1514.png"><img src="ap_lcs/lightcurve_4FGLJ0035.2p1514.png" width="200" height="155" alt="lightcurve_4FGLJ0035.2p1514.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0035.2p1514.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0035.8-0837</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0035.8-0837.png"><img src="ap_lcs/lightcurve_4FGLJ0035.8-0837.png" width="200" height="155" alt="lightcurve_4FGLJ0035.8-0837.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0035.8-0837.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0035.8+6131</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0035.8p6131.png"><img src="ap_lcs/lightcurve_4FGLJ0035.8p6131.png" width="200" height="155" alt="lightcurve_4FGLJ0035.8p6131.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0035.8p6131.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0035.9+5950</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0035.9p5950.png"><img src="ap_lcs/lightcurve_4FGLJ0035.9p5950.png" width="200" height="155" alt="lightcurve_4FGLJ0035.9p5950.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0035.9p5950.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0036.9+1832</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0036.9p1832.png"><img src="ap_lcs/lightcurve_4FGLJ0036.9p1832.png" width="200" height="155" alt="lightcurve_4FGLJ0036.9p1832.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0036.9p1832.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0037.2-2653</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0037.2-2653.png"><img src="ap_lcs/lightcurve_4FGLJ0037.2-2653.png" width="200" height="155" alt="lightcurve_4FGLJ0037.2-2653.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0037.2-2653.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0037.6+3653</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0037.6p3653.png"><img src="ap_lcs/lightcurve_4FGLJ0037.6p3653.png" width="200" height="155" alt="lightcurve_4FGLJ0037.6p3653.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0037.6p3653.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0037.8+1239</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0037.8p1239.png"><img src="ap_lcs/lightcurve_4FGLJ0037.8p1239.png" width="200" height="155" alt="lightcurve_4FGLJ0037.8p1239.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0037.8p1239.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0037.9+2612</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0037.9p2612.png"><img src="ap_lcs/lightcurve_4FGLJ0037.9p2612.png" width="200" height="155" alt="lightcurve_4FGLJ0037.9p2612.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0037.9p2612.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0038.1+0012</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0038.1p0012.png"><img src="ap_lcs/lightcurve_4FGLJ0038.1p0012.png" width="200" height="155" alt="lightcurve_4FGLJ0038.1p0012.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0038.1p0012.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0038.2-2459</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0038.2-2459.png"><img src="ap_lcs/lightcurve_4FGLJ0038.2-2459.png" width="200" height="155" alt="lightcurve_4FGLJ0038.2-2459.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0038.2-2459.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0038.7-0204</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0038.7-0204.png"><img src="ap_lcs/lightcurve_4FGLJ0038.7-0204.png" width="200" height="155" alt="lightcurve_4FGLJ0038.7-0204.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0038.7-0204.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0039.0-0946</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0039.0-0946.png"><img src="ap_lcs/lightcurve_4FGLJ0039.0-0946.png" width="200" height="155" alt="lightcurve_4FGLJ0039.0-0946.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0039.0-0946.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0039.1-2219</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0039.1-2219.png"><img src="ap_lcs/lightcurve_4FGLJ0039.1-2219.png" width="200" height="155" alt="lightcurve_4FGLJ0039.1-2219.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0039.1-2219.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0039.1+4330</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0039.1p4330.png"><img src="ap_lcs/lightcurve_4FGLJ0039.1p4330.png" width="200" height="155" alt="lightcurve_4FGLJ0039.1p4330.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0039.1p4330.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0039.1+6257</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0039.1p6257.png"><img src="ap_lcs/lightcurve_4FGLJ0039.1p6257.png" width="200" height="155" alt="lightcurve_4FGLJ0039.1p6257.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0039.1p6257.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0039.7+4203</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0039.7p4203.png"><img src="ap_lcs/lightcurve_4FGLJ0039.7p4203.png" width="200" height="155" alt="lightcurve_4FGLJ0039.7p4203.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0039.7p4203.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0040.2-2725</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0040.2-2725.png"><img src="ap_lcs/lightcurve_4FGLJ0040.2-2725.png" width="200" height="155" alt="lightcurve_4FGLJ0040.2-2725.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0040.2-2725.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0040.3+4050</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0040.3p4050.png"><img src="ap_lcs/lightcurve_4FGLJ0040.3p4050.png" width="200" height="155" alt="lightcurve_4FGLJ0040.3p4050.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0040.3p4050.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0040.4-2340</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0040.4-2340.png"><img src="ap_lcs/lightcurve_4FGLJ0040.4-2340.png" width="200" height="155" alt="lightcurve_4FGLJ0040.4-2340.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0040.4-2340.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0040.7-7157</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0040.7-7157.png"><img src="ap_lcs/lightcurve_4FGLJ0040.7-7157.png" width="200" height="155" alt="lightcurve_4FGLJ0040.7-7157.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0040.7-7157.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0040.9+3203</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0040.9p3203.png"><img src="ap_lcs/lightcurve_4FGLJ0040.9p3203.png" width="200" height="155" alt="lightcurve_4FGLJ0040.9p3203.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0040.9p3203.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0041.3+6052</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0041.3p6052.png"><img src="ap_lcs/lightcurve_4FGLJ0041.3p6052.png" width="200" height="155" alt="lightcurve_4FGLJ0041.3p6052.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0041.3p6052.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0041.4+3800</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0041.4p3800.png"><img src="ap_lcs/lightcurve_4FGLJ0041.4p3800.png" width="200" height="155" alt="lightcurve_4FGLJ0041.4p3800.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0041.4p3800.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0041.7-1607</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0041.7-1607.png"><img src="ap_lcs/lightcurve_4FGLJ0041.7-1607.png" width="200" height="155" alt="lightcurve_4FGLJ0041.7-1607.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0041.7-1607.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0041.9-4702</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0041.9-4702.png"><img src="ap_lcs/lightcurve_4FGLJ0041.9-4702.png" width="200" height="155" alt="lightcurve_4FGLJ0041.9-4702.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0041.9-4702.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0042.0+3640</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0042.0p3640.png"><img src="ap_lcs/lightcurve_4FGLJ0042.0p3640.png" width="200" height="155" alt="lightcurve_4FGLJ0042.0p3640.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0042.0p3640.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0042.2+2319</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0042.2p2319.png"><img src="ap_lcs/lightcurve_4FGLJ0042.2p2319.png" width="200" height="155" alt="lightcurve_4FGLJ0042.2p2319.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0042.2p2319.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0043.2+4114</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0043.2p4114.png"><img src="ap_lcs/lightcurve_4FGLJ0043.2p4114.png" width="200" height="155" alt="lightcurve_4FGLJ0043.2p4114.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0043.2p4114.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0043.5-0442</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0043.5-0442.png"><img src="ap_lcs/lightcurve_4FGLJ0043.5-0442.png" width="200" height="155" alt="lightcurve_4FGLJ0043.5-0442.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0043.5-0442.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0043.6+2223</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0043.6p2223.png"><img src="ap_lcs/lightcurve_4FGLJ0043.6p2223.png" width="200" height="155" alt="lightcurve_4FGLJ0043.6p2223.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0043.6p2223.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0043.7-1116</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0043.7-1116.png"><img src="ap_lcs/lightcurve_4FGLJ0043.7-1116.png" width="200" height="155" alt="lightcurve_4FGLJ0043.7-1116.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0043.7-1116.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0043.8+3425</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0043.8p3425.png"><img src="ap_lcs/lightcurve_4FGLJ0043.8p3425.png" width="200" height="155" alt="lightcurve_4FGLJ0043.8p3425.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0043.8p3425.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0044.2-8424</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0044.2-8424.png"><img src="ap_lcs/lightcurve_4FGLJ0044.2-8424.png" width="200" height="155" alt="lightcurve_4FGLJ0044.2-8424.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0044.2-8424.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0044.8+6802</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0044.8p6802.png"><img src="ap_lcs/lightcurve_4FGLJ0044.8p6802.png" width="200" height="155" alt="lightcurve_4FGLJ0044.8p6802.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0044.8p6802.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0045.1-3706</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0045.1-3706.png"><img src="ap_lcs/lightcurve_4FGLJ0045.1-3706.png" width="200" height="155" alt="lightcurve_4FGLJ0045.1-3706.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0045.1-3706.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0045.3+2128</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0045.3p2128.png"><img src="ap_lcs/lightcurve_4FGLJ0045.3p2128.png" width="200" height="155" alt="lightcurve_4FGLJ0045.3p2128.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0045.3p2128.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0045.7+1217</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0045.7p1217.png"><img src="ap_lcs/lightcurve_4FGLJ0045.7p1217.png" width="200" height="155" alt="lightcurve_4FGLJ0045.7p1217.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0045.7p1217.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0045.8-1324</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0045.8-1324.png"><img src="ap_lcs/lightcurve_4FGLJ0045.8-1324.png" width="200" height="155" alt="lightcurve_4FGLJ0045.8-1324.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0045.8-1324.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0045.9-2021</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0045.9-2021.png"><img src="ap_lcs/lightcurve_4FGLJ0045.9-2021.png" width="200" height="155" alt="lightcurve_4FGLJ0045.9-2021.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0045.9-2021.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0046.7-7048</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0046.7-7048.png"><img src="ap_lcs/lightcurve_4FGLJ0046.7-7048.png" width="200" height="155" alt="lightcurve_4FGLJ0046.7-7048.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0046.7-7048.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0047.0+5657</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0047.0p5657.png"><img src="ap_lcs/lightcurve_4FGLJ0047.0p5657.png" width="200" height="155" alt="lightcurve_4FGLJ0047.0p5657.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0047.0p5657.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0047.1-6203</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0047.1-6203.png"><img src="ap_lcs/lightcurve_4FGLJ0047.1-6203.png" width="200" height="155" alt="lightcurve_4FGLJ0047.1-6203.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0047.1-6203.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0047.3+6943</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0047.3p6943.png"><img src="ap_lcs/lightcurve_4FGLJ0047.3p6943.png" width="200" height="155" alt="lightcurve_4FGLJ0047.3p6943.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0047.3p6943.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0047.5-2517</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0047.5-2517.png"><img src="ap_lcs/lightcurve_4FGLJ0047.5-2517.png" width="200" height="155" alt="lightcurve_4FGLJ0047.5-2517.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0047.5-2517.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0047.9+2233</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0047.9p2233.png"><img src="ap_lcs/lightcurve_4FGLJ0047.9p2233.png" width="200" height="155" alt="lightcurve_4FGLJ0047.9p2233.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0047.9p2233.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0047.9+3947</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0047.9p3947.png"><img src="ap_lcs/lightcurve_4FGLJ0047.9p3947.png" width="200" height="155" alt="lightcurve_4FGLJ0047.9p3947.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0047.9p3947.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0047.9+5448</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0047.9p5448.png"><img src="ap_lcs/lightcurve_4FGLJ0047.9p5448.png" width="200" height="155" alt="lightcurve_4FGLJ0047.9p5448.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0047.9p5448.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0048.6-2427</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0048.6-2427.png"><img src="ap_lcs/lightcurve_4FGLJ0048.6-2427.png" width="200" height="155" alt="lightcurve_4FGLJ0048.6-2427.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0048.6-2427.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0048.6-6347</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0048.6-6347.png"><img src="ap_lcs/lightcurve_4FGLJ0048.6-6347.png" width="200" height="155" alt="lightcurve_4FGLJ0048.6-6347.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0048.6-6347.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0049.0+2252</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0049.0p2252.png"><img src="ap_lcs/lightcurve_4FGLJ0049.0p2252.png" width="200" height="155" alt="lightcurve_4FGLJ0049.0p2252.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0049.0p2252.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0049.1+4223</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0049.1p4223.png"><img src="ap_lcs/lightcurve_4FGLJ0049.1p4223.png" width="200" height="155" alt="lightcurve_4FGLJ0049.1p4223.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0049.1p4223.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0049.4-5402</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0049.4-5402.png"><img src="ap_lcs/lightcurve_4FGLJ0049.4-5402.png" width="200" height="155" alt="lightcurve_4FGLJ0049.4-5402.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0049.4-5402.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0049.5-4150</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0049.5-4150.png"><img src="ap_lcs/lightcurve_4FGLJ0049.5-4150.png" width="200" height="155" alt="lightcurve_4FGLJ0049.5-4150.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0049.5-4150.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0049.6-4500</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0049.6-4500.png"><img src="ap_lcs/lightcurve_4FGLJ0049.6-4500.png" width="200" height="155" alt="lightcurve_4FGLJ0049.6-4500.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0049.6-4500.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0049.7+0237</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0049.7p0237.png"><img src="ap_lcs/lightcurve_4FGLJ0049.7p0237.png" width="200" height="155" alt="lightcurve_4FGLJ0049.7p0237.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0049.7p0237.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0050.0-5736</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0050.0-5736.png"><img src="ap_lcs/lightcurve_4FGLJ0050.0-5736.png" width="200" height="155" alt="lightcurve_4FGLJ0050.0-5736.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0050.0-5736.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0050.4-0452</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0050.4-0452.png"><img src="ap_lcs/lightcurve_4FGLJ0050.4-0452.png" width="200" height="155" alt="lightcurve_4FGLJ0050.4-0452.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0050.4-0452.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0050.7-0929</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0050.7-0929.png"><img src="ap_lcs/lightcurve_4FGLJ0050.7-0929.png" width="200" height="155" alt="lightcurve_4FGLJ0050.7-0929.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0050.7-0929.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0051.1-0648</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0051.1-0648.png"><img src="ap_lcs/lightcurve_4FGLJ0051.1-0648.png" width="200" height="155" alt="lightcurve_4FGLJ0051.1-0648.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0051.1-0648.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0051.2-6242</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0051.2-6242.png"><img src="ap_lcs/lightcurve_4FGLJ0051.2-6242.png" width="200" height="155" alt="lightcurve_4FGLJ0051.2-6242.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0051.2-6242.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0051.5-4220</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0051.5-4220.png"><img src="ap_lcs/lightcurve_4FGLJ0051.5-4220.png" width="200" height="155" alt="lightcurve_4FGLJ0051.5-4220.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0051.5-4220.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0052.1+6444</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0052.1p6444.png"><img src="ap_lcs/lightcurve_4FGLJ0052.1p6444.png" width="200" height="155" alt="lightcurve_4FGLJ0052.1p6444.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0052.1p6444.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0052.9-6644</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0052.9-6644.png"><img src="ap_lcs/lightcurve_4FGLJ0052.9-6644.png" width="200" height="155" alt="lightcurve_4FGLJ0052.9-6644.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0052.9-6644.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0054.4-1503</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0054.4-1503.png"><img src="ap_lcs/lightcurve_4FGLJ0054.4-1503.png" width="200" height="155" alt="lightcurve_4FGLJ0054.4-1503.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0054.4-1503.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0054.4+8627</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0054.4p8627.png"><img src="ap_lcs/lightcurve_4FGLJ0054.4p8627.png" width="200" height="155" alt="lightcurve_4FGLJ0054.4p8627.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0054.4p8627.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0054.7-2455</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0054.7-2455.png"><img src="ap_lcs/lightcurve_4FGLJ0054.7-2455.png" width="200" height="155" alt="lightcurve_4FGLJ0054.7-2455.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0054.7-2455.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0054.8-1954</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0054.8-1954.png"><img src="ap_lcs/lightcurve_4FGLJ0054.8-1954.png" width="200" height="155" alt="lightcurve_4FGLJ0054.8-1954.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0054.8-1954.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0055.1-1219</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0055.1-1219.png"><img src="ap_lcs/lightcurve_4FGLJ0055.1-1219.png" width="200" height="155" alt="lightcurve_4FGLJ0055.1-1219.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0055.1-1219.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0055.7+4507</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0055.7p4507.png"><img src="ap_lcs/lightcurve_4FGLJ0055.7p4507.png" width="200" height="155" alt="lightcurve_4FGLJ0055.7p4507.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0055.7p4507.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0056.3-0935</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0056.3-0935.png"><img src="ap_lcs/lightcurve_4FGLJ0056.3-0935.png" width="200" height="155" alt="lightcurve_4FGLJ0056.3-0935.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0056.3-0935.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0056.4-2118</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0056.4-2118.png"><img src="ap_lcs/lightcurve_4FGLJ0056.4-2118.png" width="200" height="155" alt="lightcurve_4FGLJ0056.4-2118.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0056.4-2118.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0056.5-3936</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0056.5-3936.png"><img src="ap_lcs/lightcurve_4FGLJ0056.5-3936.png" width="200" height="155" alt="lightcurve_4FGLJ0056.5-3936.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0056.5-3936.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0056.6-4452</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0056.6-4452.png"><img src="ap_lcs/lightcurve_4FGLJ0056.6-4452.png" width="200" height="155" alt="lightcurve_4FGLJ0056.6-4452.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0056.6-4452.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0056.6-5317</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0056.6-5317.png"><img src="ap_lcs/lightcurve_4FGLJ0056.6-5317.png" width="200" height="155" alt="lightcurve_4FGLJ0056.6-5317.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0056.6-5317.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0056.8+1626</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0056.8p1626.png"><img src="ap_lcs/lightcurve_4FGLJ0056.8p1626.png" width="200" height="155" alt="lightcurve_4FGLJ0056.8p1626.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0056.8p1626.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0057.0+4101</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0057.0p4101.png"><img src="ap_lcs/lightcurve_4FGLJ0057.0p4101.png" width="200" height="155" alt="lightcurve_4FGLJ0057.0p4101.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0057.0p4101.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0057.3+2216</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0057.3p2216.png"><img src="ap_lcs/lightcurve_4FGLJ0057.3p2216.png" width="200" height="155" alt="lightcurve_4FGLJ0057.3p2216.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0057.3p2216.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0057.5+6814</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0057.5p6814.png"><img src="ap_lcs/lightcurve_4FGLJ0057.5p6814.png" width="200" height="155" alt="lightcurve_4FGLJ0057.5p6814.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0057.5p6814.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0057.7+3023</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0057.7p3023.png"><img src="ap_lcs/lightcurve_4FGLJ0057.7p3023.png" width="200" height="155" alt="lightcurve_4FGLJ0057.7p3023.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0057.7p3023.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0057.9+6326</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0057.9p6326.png"><img src="ap_lcs/lightcurve_4FGLJ0057.9p6326.png" width="200" height="155" alt="lightcurve_4FGLJ0057.9p6326.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0057.9p6326.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0058.0-0539</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0058.0-0539.png"><img src="ap_lcs/lightcurve_4FGLJ0058.0-0539.png" width="200" height="155" alt="lightcurve_4FGLJ0058.0-0539.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0058.0-0539.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0058.0-3233</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0058.0-3233.png"><img src="ap_lcs/lightcurve_4FGLJ0058.0-3233.png" width="200" height="155" alt="lightcurve_4FGLJ0058.0-3233.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0058.0-3233.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0058.0-7245e</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0058.0-7245e.png"><img src="ap_lcs/lightcurve_4FGLJ0058.0-7245e.png" width="200" height="155" alt="lightcurve_4FGLJ0058.0-7245e.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0058.0-7245e.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0058.1+6915</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0058.1p6915.png"><img src="ap_lcs/lightcurve_4FGLJ0058.1p6915.png" width="200" height="155" alt="lightcurve_4FGLJ0058.1p6915.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0058.1p6915.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0058.3-4603</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0058.3-4603.png"><img src="ap_lcs/lightcurve_4FGLJ0058.3-4603.png" width="200" height="155" alt="lightcurve_4FGLJ0058.3-4603.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0058.3-4603.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0058.3+1723</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0058.3p1723.png"><img src="ap_lcs/lightcurve_4FGLJ0058.3p1723.png" width="200" height="155" alt="lightcurve_4FGLJ0058.3p1723.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0058.3p1723.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0058.4+3315</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0058.4p3315.png"><img src="ap_lcs/lightcurve_4FGLJ0058.4p3315.png" width="200" height="155" alt="lightcurve_4FGLJ0058.4p3315.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0058.4p3315.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0058.6-1140</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0058.6-1140.png"><img src="ap_lcs/lightcurve_4FGLJ0058.6-1140.png" width="200" height="155" alt="lightcurve_4FGLJ0058.6-1140.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0058.6-1140.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0059.2+0006</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0059.2p0006.png"><img src="ap_lcs/lightcurve_4FGLJ0059.2p0006.png" width="200" height="155" alt="lightcurve_4FGLJ0059.2p0006.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0059.2p0006.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0059.3-0152</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0059.3-0152.png"><img src="ap_lcs/lightcurve_4FGLJ0059.3-0152.png" width="200" height="155" alt="lightcurve_4FGLJ0059.3-0152.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0059.3-0152.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0059.4-5654</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0059.4-5654.png"><img src="ap_lcs/lightcurve_4FGLJ0059.4-5654.png" width="200" height="155" alt="lightcurve_4FGLJ0059.4-5654.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0059.4-5654.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0059.5-3338</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0059.5-3338.png"><img src="ap_lcs/lightcurve_4FGLJ0059.5-3338.png" width="200" height="155" alt="lightcurve_4FGLJ0059.5-3338.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0059.5-3338.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0059.5-3512</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0059.5-3512.png"><img src="ap_lcs/lightcurve_4FGLJ0059.5-3512.png" width="200" height="155" alt="lightcurve_4FGLJ0059.5-3512.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0059.5-3512.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0059.7-7210</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0059.7-7210.png"><img src="ap_lcs/lightcurve_4FGLJ0059.7-7210.png" width="200" height="155" alt="lightcurve_4FGLJ0059.7-7210.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0059.7-7210.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0100.3+0745</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0100.3p0745.png"><img src="ap_lcs/lightcurve_4FGLJ0100.3p0745.png" width="200" height="155" alt="lightcurve_4FGLJ0100.3p0745.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0100.3p0745.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0101.0-0059</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0101.0-0059.png"><img src="ap_lcs/lightcurve_4FGLJ0101.0-0059.png" width="200" height="155" alt="lightcurve_4FGLJ0101.0-0059.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0101.0-0059.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0101.1-6422</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0101.1-6422.png"><img src="ap_lcs/lightcurve_4FGLJ0101.1-6422.png" width="200" height="155" alt="lightcurve_4FGLJ0101.1-6422.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0101.1-6422.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0101.7-5455</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0101.7-5455.png"><img src="ap_lcs/lightcurve_4FGLJ0101.7-5455.png" width="200" height="155" alt="lightcurve_4FGLJ0101.7-5455.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0101.7-5455.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0101.8-7543</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0101.8-7543.png"><img src="ap_lcs/lightcurve_4FGLJ0101.8-7543.png" width="200" height="155" alt="lightcurve_4FGLJ0101.8-7543.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0101.8-7543.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0102.0-6240</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0102.0-6240.png"><img src="ap_lcs/lightcurve_4FGLJ0102.0-6240.png" width="200" height="155" alt="lightcurve_4FGLJ0102.0-6240.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0102.0-6240.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0102.0+1639</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0102.0p1639.png"><img src="ap_lcs/lightcurve_4FGLJ0102.0p1639.png" width="200" height="155" alt="lightcurve_4FGLJ0102.0p1639.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0102.0p1639.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0102.1+4458</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0102.1p4458.png"><img src="ap_lcs/lightcurve_4FGLJ0102.1p4458.png" width="200" height="155" alt="lightcurve_4FGLJ0102.1p4458.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0102.1p4458.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0102.3+1000</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0102.3p1000.png"><img src="ap_lcs/lightcurve_4FGLJ0102.3p1000.png" width="200" height="155" alt="lightcurve_4FGLJ0102.3p1000.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0102.3p1000.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0102.4+0942</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0102.4p0942.png"><img src="ap_lcs/lightcurve_4FGLJ0102.4p0942.png" width="200" height="155" alt="lightcurve_4FGLJ0102.4p0942.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0102.4p0942.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0102.4+4214</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0102.4p4214.png"><img src="ap_lcs/lightcurve_4FGLJ0102.4p4214.png" width="200" height="155" alt="lightcurve_4FGLJ0102.4p4214.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0102.4p4214.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0102.6-5639</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0102.6-5639.png"><img src="ap_lcs/lightcurve_4FGLJ0102.6-5639.png" width="200" height="155" alt="lightcurve_4FGLJ0102.6-5639.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0102.6-5639.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0102.7-2001</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0102.7-2001.png"><img src="ap_lcs/lightcurve_4FGLJ0102.7-2001.png" width="200" height="155" alt="lightcurve_4FGLJ0102.7-2001.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0102.7-2001.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0102.8+4839</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0102.8p4839.png"><img src="ap_lcs/lightcurve_4FGLJ0102.8p4839.png" width="200" height="155" alt="lightcurve_4FGLJ0102.8p4839.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0102.8p4839.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0102.8+5824</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0102.8p5824.png"><img src="ap_lcs/lightcurve_4FGLJ0102.8p5824.png" width="200" height="155" alt="lightcurve_4FGLJ0102.8p5824.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0102.8p5824.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0102.9-7051</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0102.9-7051.png"><img src="ap_lcs/lightcurve_4FGLJ0102.9-7051.png" width="200" height="155" alt="lightcurve_4FGLJ0102.9-7051.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0102.9-7051.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0103.1+4954</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0103.1p4954.png"><img src="ap_lcs/lightcurve_4FGLJ0103.1p4954.png" width="200" height="155" alt="lightcurve_4FGLJ0103.1p4954.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0103.1p4954.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0103.5+1526</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0103.5p1526.png"><img src="ap_lcs/lightcurve_4FGLJ0103.5p1526.png" width="200" height="155" alt="lightcurve_4FGLJ0103.5p1526.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0103.5p1526.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0103.5+5337</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0103.5p5337.png"><img src="ap_lcs/lightcurve_4FGLJ0103.5p5337.png" width="200" height="155" alt="lightcurve_4FGLJ0103.5p5337.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0103.5p5337.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0103.8+1321</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0103.8p1321.png"><img src="ap_lcs/lightcurve_4FGLJ0103.8p1321.png" width="200" height="155" alt="lightcurve_4FGLJ0103.8p1321.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0103.8p1321.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0104.6-0818</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0104.6-0818.png"><img src="ap_lcs/lightcurve_4FGLJ0104.6-0818.png" width="200" height="155" alt="lightcurve_4FGLJ0104.6-0818.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0104.6-0818.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0104.8-2416</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0104.8-2416.png"><img src="ap_lcs/lightcurve_4FGLJ0104.8-2416.png" width="200" height="155" alt="lightcurve_4FGLJ0104.8-2416.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0104.8-2416.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0105.1+3929</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0105.1p3929.png"><img src="ap_lcs/lightcurve_4FGLJ0105.1p3929.png" width="200" height="155" alt="lightcurve_4FGLJ0105.1p3929.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0105.1p3929.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0106.1+1131</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0106.1p1131.png"><img src="ap_lcs/lightcurve_4FGLJ0106.1p1131.png" width="200" height="155" alt="lightcurve_4FGLJ0106.1p1131.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0106.1p1131.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0106.4+4855</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0106.4p4855.png"><img src="ap_lcs/lightcurve_4FGLJ0106.4p4855.png" width="200" height="155" alt="lightcurve_4FGLJ0106.4p4855.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0106.4p4855.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0106.9-4832</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0106.9-4832.png"><img src="ap_lcs/lightcurve_4FGLJ0106.9-4832.png" width="200" height="155" alt="lightcurve_4FGLJ0106.9-4832.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0106.9-4832.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0107.3-1210</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0107.3-1210.png"><img src="ap_lcs/lightcurve_4FGLJ0107.3-1210.png" width="200" height="155" alt="lightcurve_4FGLJ0107.3-1210.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0107.3-1210.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0107.4+0334</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0107.4p0334.png"><img src="ap_lcs/lightcurve_4FGLJ0107.4p0334.png" width="200" height="155" alt="lightcurve_4FGLJ0107.4p0334.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0107.4p0334.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0108.1-0039</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0108.1-0039.png"><img src="ap_lcs/lightcurve_4FGLJ0108.1-0039.png" width="200" height="155" alt="lightcurve_4FGLJ0108.1-0039.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0108.1-0039.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0108.6+0134</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0108.6p0134.png"><img src="ap_lcs/lightcurve_4FGLJ0108.6p0134.png" width="200" height="155" alt="lightcurve_4FGLJ0108.6p0134.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0108.6p0134.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0109.1+1815</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0109.1p1815.png"><img src="ap_lcs/lightcurve_4FGLJ0109.1p1815.png" width="200" height="155" alt="lightcurve_4FGLJ0109.1p1815.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0109.1p1815.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0109.3+2401</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0109.3p2401.png"><img src="ap_lcs/lightcurve_4FGLJ0109.3p2401.png" width="200" height="155" alt="lightcurve_4FGLJ0109.3p2401.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0109.3p2401.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0109.7+6133</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0109.7p6133.png"><img src="ap_lcs/lightcurve_4FGLJ0109.7p6133.png" width="200" height="155" alt="lightcurve_4FGLJ0109.7p6133.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0109.7p6133.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0110.0-4019</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0110.0-4019.png"><img src="ap_lcs/lightcurve_4FGLJ0110.0-4019.png" width="200" height="155" alt="lightcurve_4FGLJ0110.0-4019.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0110.0-4019.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0110.1+6805</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0110.1p6805.png"><img src="ap_lcs/lightcurve_4FGLJ0110.1p6805.png" width="200" height="155" alt="lightcurve_4FGLJ0110.1p6805.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0110.1p6805.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0110.2+4151</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0110.2p4151.png"><img src="ap_lcs/lightcurve_4FGLJ0110.2p4151.png" width="200" height="155" alt="lightcurve_4FGLJ0110.2p4151.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0110.2p4151.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0110.7-1254</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0110.7-1254.png"><img src="ap_lcs/lightcurve_4FGLJ0110.7-1254.png" width="200" height="155" alt="lightcurve_4FGLJ0110.7-1254.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0110.7-1254.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0110.9+4344</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0110.9p4344.png"><img src="ap_lcs/lightcurve_4FGLJ0110.9p4344.png" width="200" height="155" alt="lightcurve_4FGLJ0110.9p4344.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0110.9p4344.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0111.4+0534</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0111.4p0534.png"><img src="ap_lcs/lightcurve_4FGLJ0111.4p0534.png" width="200" height="155" alt="lightcurve_4FGLJ0111.4p0534.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0111.4p0534.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0111.5-2546</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0111.5-2546.png"><img src="ap_lcs/lightcurve_4FGLJ0111.5-2546.png" width="200" height="155" alt="lightcurve_4FGLJ0111.5-2546.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0111.5-2546.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0112.0-6634</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0112.0-6634.png"><img src="ap_lcs/lightcurve_4FGLJ0112.0-6634.png" width="200" height="155" alt="lightcurve_4FGLJ0112.0-6634.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0112.0-6634.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0112.0+3442</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0112.0p3442.png"><img src="ap_lcs/lightcurve_4FGLJ0112.0p3442.png" width="200" height="155" alt="lightcurve_4FGLJ0112.0p3442.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0112.0p3442.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0112.1-0321</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0112.1-0321.png"><img src="ap_lcs/lightcurve_4FGLJ0112.1-0321.png" width="200" height="155" alt="lightcurve_4FGLJ0112.1-0321.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0112.1-0321.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0112.1+2245</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0112.1p2245.png"><img src="ap_lcs/lightcurve_4FGLJ0112.1p2245.png" width="200" height="155" alt="lightcurve_4FGLJ0112.1p2245.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0112.1p2245.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0112.6-3158</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0112.6-3158.png"><img src="ap_lcs/lightcurve_4FGLJ0112.6-3158.png" width="200" height="155" alt="lightcurve_4FGLJ0112.6-3158.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0112.6-3158.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0112.8-7506</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0112.8-7506.png"><img src="ap_lcs/lightcurve_4FGLJ0112.8-7506.png" width="200" height="155" alt="lightcurve_4FGLJ0112.8-7506.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0112.8-7506.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0112.8+3208</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0112.8p3208.png"><img src="ap_lcs/lightcurve_4FGLJ0112.8p3208.png" width="200" height="155" alt="lightcurve_4FGLJ0112.8p3208.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0112.8p3208.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0113.1-3553</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0113.1-3553.png"><img src="ap_lcs/lightcurve_4FGLJ0113.1-3553.png" width="200" height="155" alt="lightcurve_4FGLJ0113.1-3553.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0113.1-3553.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0113.4+4948</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0113.4p4948.png"><img src="ap_lcs/lightcurve_4FGLJ0113.4p4948.png" width="200" height="155" alt="lightcurve_4FGLJ0113.4p4948.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0113.4p4948.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0113.7+0225</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0113.7p0225.png"><img src="ap_lcs/lightcurve_4FGLJ0113.7p0225.png" width="200" height="155" alt="lightcurve_4FGLJ0113.7p0225.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0113.7p0225.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0114.0+6418</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0114.0p6418.png"><img src="ap_lcs/lightcurve_4FGLJ0114.0p6418.png" width="200" height="155" alt="lightcurve_4FGLJ0114.0p6418.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0114.0p6418.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0114.8+1326</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0114.8p1326.png"><img src="ap_lcs/lightcurve_4FGLJ0114.8p1326.png" width="200" height="155" alt="lightcurve_4FGLJ0114.8p1326.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0114.8p1326.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0114.9-3400</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0114.9-3400.png"><img src="ap_lcs/lightcurve_4FGLJ0114.9-3400.png" width="200" height="155" alt="lightcurve_4FGLJ0114.9-3400.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0114.9-3400.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0115.1-0129</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0115.1-0129.png"><img src="ap_lcs/lightcurve_4FGLJ0115.1-0129.png" width="200" height="155" alt="lightcurve_4FGLJ0115.1-0129.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0115.1-0129.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0115.1+2622</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0115.1p2622.png"><img src="ap_lcs/lightcurve_4FGLJ0115.1p2622.png" width="200" height="155" alt="lightcurve_4FGLJ0115.1p2622.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0115.1p2622.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0115.4-2917</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0115.4-2917.png"><img src="ap_lcs/lightcurve_4FGLJ0115.4-2917.png" width="200" height="155" alt="lightcurve_4FGLJ0115.4-2917.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0115.4-2917.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0115.6+0356</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0115.6p0356.png"><img src="ap_lcs/lightcurve_4FGLJ0115.6p0356.png" width="200" height="155" alt="lightcurve_4FGLJ0115.6p0356.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0115.6p0356.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0115.8+2519</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0115.8p2519.png"><img src="ap_lcs/lightcurve_4FGLJ0115.8p2519.png" width="200" height="155" alt="lightcurve_4FGLJ0115.8p2519.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0115.8p2519.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0116.0-1136</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0116.0-1136.png"><img src="ap_lcs/lightcurve_4FGLJ0116.0-1136.png" width="200" height="155" alt="lightcurve_4FGLJ0116.0-1136.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0116.0-1136.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0116.0-2745</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0116.0-2745.png"><img src="ap_lcs/lightcurve_4FGLJ0116.0-2745.png" width="200" height="155" alt="lightcurve_4FGLJ0116.0-2745.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0116.0-2745.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0116.2-6153</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0116.2-6153.png"><img src="ap_lcs/lightcurve_4FGLJ0116.2-6153.png" width="200" height="155" alt="lightcurve_4FGLJ0116.2-6153.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0116.2-6153.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0116.5-2812</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0116.5-2812.png"><img src="ap_lcs/lightcurve_4FGLJ0116.5-2812.png" width="200" height="155" alt="lightcurve_4FGLJ0116.5-2812.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0116.5-2812.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0116.5-3046</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0116.5-3046.png"><img src="ap_lcs/lightcurve_4FGLJ0116.5-3046.png" width="200" height="155" alt="lightcurve_4FGLJ0116.5-3046.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0116.5-3046.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0116.8+6914</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0116.8p6914.png"><img src="ap_lcs/lightcurve_4FGLJ0116.8p6914.png" width="200" height="155" alt="lightcurve_4FGLJ0116.8p6914.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0116.8p6914.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0117.5-2442</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0117.5-2442.png"><img src="ap_lcs/lightcurve_4FGLJ0117.5-2442.png" width="200" height="155" alt="lightcurve_4FGLJ0117.5-2442.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0117.5-2442.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0117.8-2109</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0117.8-2109.png"><img src="ap_lcs/lightcurve_4FGLJ0117.8-2109.png" width="200" height="155" alt="lightcurve_4FGLJ0117.8-2109.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0117.8-2109.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0118.3-6008</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0118.3-6008.png"><img src="ap_lcs/lightcurve_4FGLJ0118.3-6008.png" width="200" height="155" alt="lightcurve_4FGLJ0118.3-6008.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0118.3-6008.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0118.7-0848</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0118.7-0848.png"><img src="ap_lcs/lightcurve_4FGLJ0118.7-0848.png" width="200" height="155" alt="lightcurve_4FGLJ0118.7-0848.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0118.7-0848.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0118.9-2141</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0118.9-2141.png"><img src="ap_lcs/lightcurve_4FGLJ0118.9-2141.png" width="200" height="155" alt="lightcurve_4FGLJ0118.9-2141.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0118.9-2141.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0119.0-1458</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0119.0-1458.png"><img src="ap_lcs/lightcurve_4FGLJ0119.0-1458.png" width="200" height="155" alt="lightcurve_4FGLJ0119.0-1458.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0119.0-1458.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0119.4-5354</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0119.4-5354.png"><img src="ap_lcs/lightcurve_4FGLJ0119.4-5354.png" width="200" height="155" alt="lightcurve_4FGLJ0119.4-5354.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0119.4-5354.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0119.6+4158</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0119.6p4158.png"><img src="ap_lcs/lightcurve_4FGLJ0119.6p4158.png" width="200" height="155" alt="lightcurve_4FGLJ0119.6p4158.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0119.6p4158.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0119.9+4053</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0119.9p4053.png"><img src="ap_lcs/lightcurve_4FGLJ0119.9p4053.png" width="200" height="155" alt="lightcurve_4FGLJ0119.9p4053.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0119.9p4053.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0120.1+0505</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0120.1p0505.png"><img src="ap_lcs/lightcurve_4FGLJ0120.1p0505.png" width="200" height="155" alt="lightcurve_4FGLJ0120.1p0505.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0120.1p0505.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0120.2-7944</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0120.2-7944.png"><img src="ap_lcs/lightcurve_4FGLJ0120.2-7944.png" width="200" height="155" alt="lightcurve_4FGLJ0120.2-7944.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0120.2-7944.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0120.4-2701</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0120.4-2701.png"><img src="ap_lcs/lightcurve_4FGLJ0120.4-2701.png" width="200" height="155" alt="lightcurve_4FGLJ0120.4-2701.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0120.4-2701.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0121.7+5153</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0121.7p5153.png"><img src="ap_lcs/lightcurve_4FGLJ0121.7p5153.png" width="200" height="155" alt="lightcurve_4FGLJ0121.7p5153.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0121.7p5153.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0121.8-3916</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0121.8-3916.png"><img src="ap_lcs/lightcurve_4FGLJ0121.8-3916.png" width="200" height="155" alt="lightcurve_4FGLJ0121.8-3916.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0121.8-3916.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0122.1-3004</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0122.1-3004.png"><img src="ap_lcs/lightcurve_4FGLJ0122.1-3004.png" width="200" height="155" alt="lightcurve_4FGLJ0122.1-3004.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0122.1-3004.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0122.4+1034</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0122.4p1034.png"><img src="ap_lcs/lightcurve_4FGLJ0122.4p1034.png" width="200" height="155" alt="lightcurve_4FGLJ0122.4p1034.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0122.4p1034.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0123.1+3421</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0123.1p3421.png"><img src="ap_lcs/lightcurve_4FGLJ0123.1p3421.png" width="200" height="155" alt="lightcurve_4FGLJ0123.1p3421.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0123.1p3421.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0123.7-2311</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0123.7-2311.png"><img src="ap_lcs/lightcurve_4FGLJ0123.7-2311.png" width="200" height="155" alt="lightcurve_4FGLJ0123.7-2311.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0123.7-2311.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0124.8-0625</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0124.8-0625.png"><img src="ap_lcs/lightcurve_4FGLJ0124.8-0625.png" width="200" height="155" alt="lightcurve_4FGLJ0124.8-0625.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0124.8-0625.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0125.3-2548</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0125.3-2548.png"><img src="ap_lcs/lightcurve_4FGLJ0125.3-2548.png" width="200" height="155" alt="lightcurve_4FGLJ0125.3-2548.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0125.3-2548.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0125.3+6820</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0125.3p6820.png"><img src="ap_lcs/lightcurve_4FGLJ0125.3p6820.png" width="200" height="155" alt="lightcurve_4FGLJ0125.3p6820.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0125.3p6820.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0125.4+3200</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0125.4p3200.png"><img src="ap_lcs/lightcurve_4FGLJ0125.4p3200.png" width="200" height="155" alt="lightcurve_4FGLJ0125.4p3200.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0125.4p3200.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0125.7-0015</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0125.7-0015.png"><img src="ap_lcs/lightcurve_4FGLJ0125.7-0015.png" width="200" height="155" alt="lightcurve_4FGLJ0125.7-0015.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0125.7-0015.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0125.9-6303</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0125.9-6303.png"><img src="ap_lcs/lightcurve_4FGLJ0125.9-6303.png" width="200" height="155" alt="lightcurve_4FGLJ0125.9-6303.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0125.9-6303.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0126.0-2221</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0126.0-2221.png"><img src="ap_lcs/lightcurve_4FGLJ0126.0-2221.png" width="200" height="155" alt="lightcurve_4FGLJ0126.0-2221.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0126.0-2221.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0126.3-6746</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0126.3-6746.png"><img src="ap_lcs/lightcurve_4FGLJ0126.3-6746.png" width="200" height="155" alt="lightcurve_4FGLJ0126.3-6746.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0126.3-6746.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0126.5-1553</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0126.5-1553.png"><img src="ap_lcs/lightcurve_4FGLJ0126.5-1553.png" width="200" height="155" alt="lightcurve_4FGLJ0126.5-1553.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0126.5-1553.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0126.8+2412</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0126.8p2412.png"><img src="ap_lcs/lightcurve_4FGLJ0126.8p2412.png" width="200" height="155" alt="lightcurve_4FGLJ0126.8p2412.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0126.8p2412.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0127.1+3310</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0127.1p3310.png"><img src="ap_lcs/lightcurve_4FGLJ0127.1p3310.png" width="200" height="155" alt="lightcurve_4FGLJ0127.1p3310.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0127.1p3310.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0127.2-0819</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0127.2-0819.png"><img src="ap_lcs/lightcurve_4FGLJ0127.2-0819.png" width="200" height="155" alt="lightcurve_4FGLJ0127.2-0819.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0127.2-0819.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0127.2+0324</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0127.2p0324.png"><img src="ap_lcs/lightcurve_4FGLJ0127.2p0324.png" width="200" height="155" alt="lightcurve_4FGLJ0127.2p0324.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0127.2p0324.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0127.4-4813</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0127.4-4813.png"><img src="ap_lcs/lightcurve_4FGLJ0127.4-4813.png" width="200" height="155" alt="lightcurve_4FGLJ0127.4-4813.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0127.4-4813.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0127.9+4857</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0127.9p4857.png"><img src="ap_lcs/lightcurve_4FGLJ0127.9p4857.png" width="200" height="155" alt="lightcurve_4FGLJ0127.9p4857.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0127.9p4857.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0128.1+7542</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0128.1p7542.png"><img src="ap_lcs/lightcurve_4FGLJ0128.1p7542.png" width="200" height="155" alt="lightcurve_4FGLJ0128.1p7542.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0128.1p7542.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0128.2+4400</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0128.2p4400.png"><img src="ap_lcs/lightcurve_4FGLJ0128.2p4400.png" width="200" height="155" alt="lightcurve_4FGLJ0128.2p4400.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0128.2p4400.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0128.3+5710</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0128.3p5710.png"><img src="ap_lcs/lightcurve_4FGLJ0128.3p5710.png" width="200" height="155" alt="lightcurve_4FGLJ0128.3p5710.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0128.3p5710.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0128.5+4440</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0128.5p4440.png"><img src="ap_lcs/lightcurve_4FGLJ0128.5p4440.png" width="200" height="155" alt="lightcurve_4FGLJ0128.5p4440.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0128.5p4440.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0129.0+6312</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0129.0p6312.png"><img src="ap_lcs/lightcurve_4FGLJ0129.0p6312.png" width="200" height="155" alt="lightcurve_4FGLJ0129.0p6312.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0129.0p6312.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0129.7+3436</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0129.7p3436.png"><img src="ap_lcs/lightcurve_4FGLJ0129.7p3436.png" width="200" height="155" alt="lightcurve_4FGLJ0129.7p3436.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0129.7p3436.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0129.8+1440</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0129.8p1440.png"><img src="ap_lcs/lightcurve_4FGLJ0129.8p1440.png" width="200" height="155" alt="lightcurve_4FGLJ0129.8p1440.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0129.8p1440.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0130.4-2129</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0130.4-2129.png"><img src="ap_lcs/lightcurve_4FGLJ0130.4-2129.png" width="200" height="155" alt="lightcurve_4FGLJ0130.4-2129.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0130.4-2129.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0130.6+1844</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0130.6p1844.png"><img src="ap_lcs/lightcurve_4FGLJ0130.6p1844.png" width="200" height="155" alt="lightcurve_4FGLJ0130.6p1844.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0130.6p1844.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0131.1+6120</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0131.1p6120.png"><img src="ap_lcs/lightcurve_4FGLJ0131.1p6120.png" width="200" height="155" alt="lightcurve_4FGLJ0131.1p6120.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0131.1p6120.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0131.2+5547</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0131.2p5547.png"><img src="ap_lcs/lightcurve_4FGLJ0131.2p5547.png" width="200" height="155" alt="lightcurve_4FGLJ0131.2p5547.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0131.2p5547.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0131.7-5346</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0131.7-5346.png"><img src="ap_lcs/lightcurve_4FGLJ0131.7-5346.png" width="200" height="155" alt="lightcurve_4FGLJ0131.7-5346.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0131.7-5346.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0132.1-0956</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0132.1-0956.png"><img src="ap_lcs/lightcurve_4FGLJ0132.1-0956.png" width="200" height="155" alt="lightcurve_4FGLJ0132.1-0956.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0132.1-0956.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0132.7-0804</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0132.7-0804.png"><img src="ap_lcs/lightcurve_4FGLJ0132.7-0804.png" width="200" height="155" alt="lightcurve_4FGLJ0132.7-0804.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0132.7-0804.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0132.7-1654</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0132.7-1654.png"><img src="ap_lcs/lightcurve_4FGLJ0132.7-1654.png" width="200" height="155" alt="lightcurve_4FGLJ0132.7-1654.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0132.7-1654.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0132.8-4413</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0132.8-4413.png"><img src="ap_lcs/lightcurve_4FGLJ0132.8-4413.png" width="200" height="155" alt="lightcurve_4FGLJ0132.8-4413.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0132.8-4413.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0132.8+4324</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0132.8p4324.png"><img src="ap_lcs/lightcurve_4FGLJ0132.8p4324.png" width="200" height="155" alt="lightcurve_4FGLJ0132.8p4324.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0132.8p4324.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0133.0+5931</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0133.0p5931.png"><img src="ap_lcs/lightcurve_4FGLJ0133.0p5931.png" width="200" height="155" alt="lightcurve_4FGLJ0133.0p5931.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0133.0p5931.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0133.1-5201</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0133.1-5201.png"><img src="ap_lcs/lightcurve_4FGLJ0133.1-5201.png" width="200" height="155" alt="lightcurve_4FGLJ0133.1-5201.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0133.1-5201.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0133.2-4533</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0133.2-4533.png"><img src="ap_lcs/lightcurve_4FGLJ0133.2-4533.png" width="200" height="155" alt="lightcurve_4FGLJ0133.2-4533.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0133.2-4533.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0134.3-3842</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0134.3-3842.png"><img src="ap_lcs/lightcurve_4FGLJ0134.3-3842.png" width="200" height="155" alt="lightcurve_4FGLJ0134.3-3842.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0134.3-3842.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0134.5+2637</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0134.5p2637.png"><img src="ap_lcs/lightcurve_4FGLJ0134.5p2637.png" width="200" height="155" alt="lightcurve_4FGLJ0134.5p2637.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0134.5p2637.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0135.0+5338</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0135.0p5338.png"><img src="ap_lcs/lightcurve_4FGLJ0135.0p5338.png" width="200" height="155" alt="lightcurve_4FGLJ0135.0p5338.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0135.0p5338.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0135.1+0255</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0135.1p0255.png"><img src="ap_lcs/lightcurve_4FGLJ0135.1p0255.png" width="200" height="155" alt="lightcurve_4FGLJ0135.1p0255.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0135.1p0255.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0136.5+3906</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0136.5p3906.png"><img src="ap_lcs/lightcurve_4FGLJ0136.5p3906.png" width="200" height="155" alt="lightcurve_4FGLJ0136.5p3906.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0136.5p3906.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0137.0+4751</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0137.0p4751.png"><img src="ap_lcs/lightcurve_4FGLJ0137.0p4751.png" width="200" height="155" alt="lightcurve_4FGLJ0137.0p4751.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0137.0p4751.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0137.3-3239</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0137.3-3239.png"><img src="ap_lcs/lightcurve_4FGLJ0137.3-3239.png" width="200" height="155" alt="lightcurve_4FGLJ0137.3-3239.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0137.3-3239.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0137.6-2430</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0137.6-2430.png"><img src="ap_lcs/lightcurve_4FGLJ0137.6-2430.png" width="200" height="155" alt="lightcurve_4FGLJ0137.6-2430.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0137.6-2430.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0137.9+5814</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0137.9p5814.png"><img src="ap_lcs/lightcurve_4FGLJ0137.9p5814.png" width="200" height="155" alt="lightcurve_4FGLJ0137.9p5814.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0137.9p5814.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0138.0+2247</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0138.0p2247.png"><img src="ap_lcs/lightcurve_4FGLJ0138.0p2247.png" width="200" height="155" alt="lightcurve_4FGLJ0138.0p2247.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0138.0p2247.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0138.5-4613</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0138.5-4613.png"><img src="ap_lcs/lightcurve_4FGLJ0138.5-4613.png" width="200" height="155" alt="lightcurve_4FGLJ0138.5-4613.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0138.5-4613.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0138.5+0300</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0138.5p0300.png"><img src="ap_lcs/lightcurve_4FGLJ0138.5p0300.png" width="200" height="155" alt="lightcurve_4FGLJ0138.5p0300.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0138.5p0300.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0138.6+6821</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0138.6p6821.png"><img src="ap_lcs/lightcurve_4FGLJ0138.6p6821.png" width="200" height="155" alt="lightcurve_4FGLJ0138.6p6821.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0138.6p6821.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0139.0+2601</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0139.0p2601.png"><img src="ap_lcs/lightcurve_4FGLJ0139.0p2601.png" width="200" height="155" alt="lightcurve_4FGLJ0139.0p2601.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0139.0p2601.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0139.5-2228</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0139.5-2228.png"><img src="ap_lcs/lightcurve_4FGLJ0139.5-2228.png" width="200" height="155" alt="lightcurve_4FGLJ0139.5-2228.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0139.5-2228.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0140.3+7054</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0140.3p7054.png"><img src="ap_lcs/lightcurve_4FGLJ0140.3p7054.png" width="200" height="155" alt="lightcurve_4FGLJ0140.3p7054.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0140.3p7054.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0140.5-4730</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0140.5-4730.png"><img src="ap_lcs/lightcurve_4FGLJ0140.5-4730.png" width="200" height="155" alt="lightcurve_4FGLJ0140.5-4730.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0140.5-4730.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0140.6-0758</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0140.6-0758.png"><img src="ap_lcs/lightcurve_4FGLJ0140.6-0758.png" width="200" height="155" alt="lightcurve_4FGLJ0140.6-0758.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0140.6-0758.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0140.6+8736</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0140.6p8736.png"><img src="ap_lcs/lightcurve_4FGLJ0140.6p8736.png" width="200" height="155" alt="lightcurve_4FGLJ0140.6p8736.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0140.6p8736.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0141.4-0928</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0141.4-0928.png"><img src="ap_lcs/lightcurve_4FGLJ0141.4-0928.png" width="200" height="155" alt="lightcurve_4FGLJ0141.4-0928.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0141.4-0928.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0142.5+6650</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0142.5p6650.png"><img src="ap_lcs/lightcurve_4FGLJ0142.5p6650.png" width="200" height="155" alt="lightcurve_4FGLJ0142.5p6650.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0142.5p6650.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0142.7-0543</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0142.7-0543.png"><img src="ap_lcs/lightcurve_4FGLJ0142.7-0543.png" width="200" height="155" alt="lightcurve_4FGLJ0142.7-0543.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0142.7-0543.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0143.1-3622</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0143.1-3622.png"><img src="ap_lcs/lightcurve_4FGLJ0143.1-3622.png" width="200" height="155" alt="lightcurve_4FGLJ0143.1-3622.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0143.1-3622.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0143.5-3156</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0143.5-3156.png"><img src="ap_lcs/lightcurve_4FGLJ0143.5-3156.png" width="200" height="155" alt="lightcurve_4FGLJ0143.5-3156.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0143.5-3156.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0143.7-5846</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0143.7-5846.png"><img src="ap_lcs/lightcurve_4FGLJ0143.7-5846.png" width="200" height="155" alt="lightcurve_4FGLJ0143.7-5846.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0143.7-5846.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0144.3+5959</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0144.3p5959.png"><img src="ap_lcs/lightcurve_4FGLJ0144.3p5959.png" width="200" height="155" alt="lightcurve_4FGLJ0144.3p5959.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0144.3p5959.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0144.6+2705</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0144.6p2705.png"><img src="ap_lcs/lightcurve_4FGLJ0144.6p2705.png" width="200" height="155" alt="lightcurve_4FGLJ0144.6p2705.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0144.6p2705.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0145.0-2732</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0145.0-2732.png"><img src="ap_lcs/lightcurve_4FGLJ0145.0-2732.png" width="200" height="155" alt="lightcurve_4FGLJ0145.0-2732.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0145.0-2732.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0145.9+2319</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0145.9p2319.png"><img src="ap_lcs/lightcurve_4FGLJ0145.9p2319.png" width="200" height="155" alt="lightcurve_4FGLJ0145.9p2319.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0145.9p2319.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0146.0-6746</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0146.0-6746.png"><img src="ap_lcs/lightcurve_4FGLJ0146.0-6746.png" width="200" height="155" alt="lightcurve_4FGLJ0146.0-6746.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0146.0-6746.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0146.3+4606</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0146.3p4606.png"><img src="ap_lcs/lightcurve_4FGLJ0146.3p4606.png" width="200" height="155" alt="lightcurve_4FGLJ0146.3p4606.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0146.3p4606.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0146.9-5202</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0146.9-5202.png"><img src="ap_lcs/lightcurve_4FGLJ0146.9-5202.png" width="200" height="155" alt="lightcurve_4FGLJ0146.9-5202.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0146.9-5202.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0147.7-1321</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0147.7-1321.png"><img src="ap_lcs/lightcurve_4FGLJ0147.7-1321.png" width="200" height="155" alt="lightcurve_4FGLJ0147.7-1321.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0147.7-1321.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0148.2+5201</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0148.2p5201.png"><img src="ap_lcs/lightcurve_4FGLJ0148.2p5201.png" width="200" height="155" alt="lightcurve_4FGLJ0148.2p5201.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0148.2p5201.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0148.6+0127</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0148.6p0127.png"><img src="ap_lcs/lightcurve_4FGLJ0148.6p0127.png" width="200" height="155" alt="lightcurve_4FGLJ0148.6p0127.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0148.6p0127.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0149.6-0734</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0149.6-0734.png"><img src="ap_lcs/lightcurve_4FGLJ0149.6-0734.png" width="200" height="155" alt="lightcurve_4FGLJ0149.6-0734.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0149.6-0734.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0150.4+4848</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0150.4p4848.png"><img src="ap_lcs/lightcurve_4FGLJ0150.4p4848.png" width="200" height="155" alt="lightcurve_4FGLJ0150.4p4848.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0150.4p4848.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0150.6-5448</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0150.6-5448.png"><img src="ap_lcs/lightcurve_4FGLJ0150.6-5448.png" width="200" height="155" alt="lightcurve_4FGLJ0150.6-5448.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0150.6-5448.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0150.9+1230</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0150.9p1230.png"><img src="ap_lcs/lightcurve_4FGLJ0150.9p1230.png" width="200" height="155" alt="lightcurve_4FGLJ0150.9p1230.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0150.9p1230.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0151.0+0539</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0151.0p0539.png"><img src="ap_lcs/lightcurve_4FGLJ0151.0p0539.png" width="200" height="155" alt="lightcurve_4FGLJ0151.0p0539.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0151.0p0539.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0151.3+8601</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0151.3p8601.png"><img src="ap_lcs/lightcurve_4FGLJ0151.3p8601.png" width="200" height="155" alt="lightcurve_4FGLJ0151.3p8601.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0151.3p8601.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0151.4-3607</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0151.4-3607.png"><img src="ap_lcs/lightcurve_4FGLJ0151.4-3607.png" width="200" height="155" alt="lightcurve_4FGLJ0151.4-3607.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0151.4-3607.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0151.7+5455</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0151.7p5455.png"><img src="ap_lcs/lightcurve_4FGLJ0151.7p5455.png" width="200" height="155" alt="lightcurve_4FGLJ0151.7p5455.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0151.7p5455.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0152.2+2206</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0152.2p2206.png"><img src="ap_lcs/lightcurve_4FGLJ0152.2p2206.png" width="200" height="155" alt="lightcurve_4FGLJ0152.2p2206.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0152.2p2206.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0152.2+3714</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0152.2p3714.png"><img src="ap_lcs/lightcurve_4FGLJ0152.2p3714.png" width="200" height="155" alt="lightcurve_4FGLJ0152.2p3714.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0152.2p3714.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0152.6+0147</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0152.6p0147.png"><img src="ap_lcs/lightcurve_4FGLJ0152.6p0147.png" width="200" height="155" alt="lightcurve_4FGLJ0152.6p0147.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0152.6p0147.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0153.0+7517</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0153.0p7517.png"><img src="ap_lcs/lightcurve_4FGLJ0153.0p7517.png" width="200" height="155" alt="lightcurve_4FGLJ0153.0p7517.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0153.0p7517.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0153.3+5416</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0153.3p5416.png"><img src="ap_lcs/lightcurve_4FGLJ0153.3p5416.png" width="200" height="155" alt="lightcurve_4FGLJ0153.3p5416.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0153.3p5416.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0153.4+7114</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0153.4p7114.png"><img src="ap_lcs/lightcurve_4FGLJ0153.4p7114.png" width="200" height="155" alt="lightcurve_4FGLJ0153.4p7114.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0153.4p7114.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0153.5-5107</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0153.5-5107.png"><img src="ap_lcs/lightcurve_4FGLJ0153.5-5107.png" width="200" height="155" alt="lightcurve_4FGLJ0153.5-5107.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0153.5-5107.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0153.9+0823</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0153.9p0823.png"><img src="ap_lcs/lightcurve_4FGLJ0153.9p0823.png" width="200" height="155" alt="lightcurve_4FGLJ0153.9p0823.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0153.9p0823.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0154.3-0236</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0154.3-0236.png"><img src="ap_lcs/lightcurve_4FGLJ0154.3-0236.png" width="200" height="155" alt="lightcurve_4FGLJ0154.3-0236.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0154.3-0236.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0154.6+0051</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0154.6p0051.png"><img src="ap_lcs/lightcurve_4FGLJ0154.6p0051.png" width="200" height="155" alt="lightcurve_4FGLJ0154.6p0051.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0154.6p0051.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0155.0+4433</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0155.0p4433.png"><img src="ap_lcs/lightcurve_4FGLJ0155.0p4433.png" width="200" height="155" alt="lightcurve_4FGLJ0155.0p4433.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0155.0p4433.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0155.4-0625</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0155.4-0625.png"><img src="ap_lcs/lightcurve_4FGLJ0155.4-0625.png" width="200" height="155" alt="lightcurve_4FGLJ0155.4-0625.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0155.4-0625.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0156.1+1502</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0156.1p1502.png"><img src="ap_lcs/lightcurve_4FGLJ0156.1p1502.png" width="200" height="155" alt="lightcurve_4FGLJ0156.1p1502.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0156.1p1502.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0156.3-2420</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0156.3-2420.png"><img src="ap_lcs/lightcurve_4FGLJ0156.3-2420.png" width="200" height="155" alt="lightcurve_4FGLJ0156.3-2420.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0156.3-2420.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0156.5+3914</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0156.5p3914.png"><img src="ap_lcs/lightcurve_4FGLJ0156.5p3914.png" width="200" height="155" alt="lightcurve_4FGLJ0156.5p3914.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0156.5p3914.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0156.6-1758</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0156.6-1758.png"><img src="ap_lcs/lightcurve_4FGLJ0156.6-1758.png" width="200" height="155" alt="lightcurve_4FGLJ0156.6-1758.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0156.6-1758.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0156.8-4744</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0156.8-4744.png"><img src="ap_lcs/lightcurve_4FGLJ0156.8-4744.png" width="200" height="155" alt="lightcurve_4FGLJ0156.8-4744.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0156.8-4744.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0156.9-5301</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0156.9-5301.png"><img src="ap_lcs/lightcurve_4FGLJ0156.9-5301.png" width="200" height="155" alt="lightcurve_4FGLJ0156.9-5301.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0156.9-5301.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0156.9+4648</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0156.9p4648.png"><img src="ap_lcs/lightcurve_4FGLJ0156.9p4648.png" width="200" height="155" alt="lightcurve_4FGLJ0156.9p4648.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0156.9p4648.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0157.7-4614</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0157.7-4614.png"><img src="ap_lcs/lightcurve_4FGLJ0157.7-4614.png" width="200" height="155" alt="lightcurve_4FGLJ0157.7-4614.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0157.7-4614.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0158.4+1230</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0158.4p1230.png"><img src="ap_lcs/lightcurve_4FGLJ0158.4p1230.png" width="200" height="155" alt="lightcurve_4FGLJ0158.4p1230.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0158.4p1230.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0158.5-3932</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0158.5-3932.png"><img src="ap_lcs/lightcurve_4FGLJ0158.5-3932.png" width="200" height="155" alt="lightcurve_4FGLJ0158.5-3932.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0158.5-3932.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0158.8+0101</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0158.8p0101.png"><img src="ap_lcs/lightcurve_4FGLJ0158.8p0101.png" width="200" height="155" alt="lightcurve_4FGLJ0158.8p0101.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0158.8p0101.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0159.0+3313</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0159.0p3313.png"><img src="ap_lcs/lightcurve_4FGLJ0159.0p3313.png" width="200" height="155" alt="lightcurve_4FGLJ0159.0p3313.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0159.0p3313.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0159.3-4523</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0159.3-4523.png"><img src="ap_lcs/lightcurve_4FGLJ0159.3-4523.png" width="200" height="155" alt="lightcurve_4FGLJ0159.3-4523.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0159.3-4523.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0159.5+1046</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0159.5p1046.png"><img src="ap_lcs/lightcurve_4FGLJ0159.5p1046.png" width="200" height="155" alt="lightcurve_4FGLJ0159.5p1046.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0159.5p1046.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0159.7-2740</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0159.7-2740.png"><img src="ap_lcs/lightcurve_4FGLJ0159.7-2740.png" width="200" height="155" alt="lightcurve_4FGLJ0159.7-2740.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0159.7-2740.dmp1.out">Data</a>
</div>
<div style="text-align: center; padding: 1em;">
<b>4FGLJ0159.8-2234</b><br>
<a href="ap_lcs/lightcurve_4FGLJ0159.8-2234.png"><img src="ap_lcs/lightcurve_4FGLJ0159.8-2234.png" width="200" height="155" alt="lightcurve_4FGLJ0159.8-2234.png"></a><br>
<a href="ap_lcs/lc_4FGLJ0159.8-2234.dmp1.out">Data</a>
</div>
</div>
<p style="text-align: center; clear: both;"><b>RA Range:</b><br><a href="ap_lcs.php?ra=00-01">00-01</a> | <a href="ap_lcs.php?ra=02-03">02-03</a> | <a href="ap_lcs.php?ra=04-05">04-05</a> | <a href="ap_lcs.php?ra=06-07">06-07</a> | <a href="ap_lcs.php?ra=08-09">08-09</a> | <a href="ap_lcs.php?ra=10-11">10-11</a> | <a href="ap_lcs.php?ra=12-13">12-13</a> | <a href="ap_lcs.php?ra=14-15">14-15</a> | <a href="ap_lcs.php?ra=16-17">16-17</a> | <a href="ap_lcs.php?ra=18-19">18-19</a> | <a href="ap_lcs.php?ra=20-21">20-21</a> | <a href="ap_lcs.php?ra=22-23">22-23</a></p>
</div>
<!-- End Section Wrapper -->


  </main>
  <!------------ shared footer ------------>
<footer class="usa-footer usa-footer--slim">
	<div class="usa-footer__primary-section usa-dark-background bg-ink">
		<div class="usa-footer__primary-container grid-row">
			<div class="mobile-lg:grid-col">
                                  
<div class="text-center margin-top-2 text-base-lightest">
  <span>Connect with us via </span>
  <a class="usa-link" href="mailto:fermihelp@athena.gsfc.nasa.gov">email</a>

  <!--a class="usa-link" href="mailto:fermihelp@athena.gsfc.nasa.gov">fermihelp@athena.gsfc.nasa.gov</a-->
  <span> or </span>
  <a class="usa-link" href="http://heasarc.gsfc.nasa.gov/cgi-bin/Feedback">webform</a>.
</div>


			</div>
		</div>

	</div>
  <div class="usa-dark-background bg-ink padding-bottom-2">
    <div class="grid-container padding-2 font-sans-2xs">
      <div class="usa-footer__logo grid-row">
        <div class="grid-col-2 display-none tablet:display-block">
          <img class="width-12" src="/inc/img/nasa-logo.svg" alt="NASA logo" />
        </div>
        <div class="grid-col-12 desktop:grid-col-2">
          <ul class="usa-list usa-list--unstyled">
            <li><a class="usa-link--external" href="https://www.nasa.gov/about/">About NASA</a></li>
            <li><a class="usa-link--external" href="https://www.nasa.gov/accessibility/">Accessibility</a></li>
					  <li><a class="usa-link--external" href="https://www.nasa.gov/foia/">FOIA</a></li>
          </ul>
        </div>
        <div class="grid-col-12 desktop:grid-col-3">
          <ul class="usa-list usa-list--unstyled text-base-lightest">
            <li>
              <a class="usa-link--external" href="https://www.nasa.gov/odeo/no-fear-act/">No FEAR Act</a>
            </li>
            <li>
              <a class="usa-link--external" href="https://www.nasa.gov/privacy/">Privacy Policy</a>
            </li>
            <li>
              <a class="usa-link--external" href="https://www.nasa.gov/vulnerability-disclosure-policy/">Vulnerability Disclosure Policy</a>
            </li>
          </ul>
        </div>
        <div class="grid-col-12 desktop:grid-col-1"></div>
        <div class="grid-col-12 desktop:grid-col-4">
          <ul class="usa-list usa-list--unstyled text-base-lightest">
            <li>
              Responsible Official: <a class="usa-link" href="mailto:ryan.smallcomb@nasa.gov">Elizabeth Hays</a>
            </li>
            <li>
              Site Editor: <a class="usa-link" href="mailto:jd.myers@nasa.gov">J.D. Myers</a>
            </li>
            <li>
              Last Updated: <!--#config timefmt="%e-%b-%Y" --><!--#echo var="LAST_MODIFIED" -->
            </li>
          </ul>
        </div>
      </div>
      <div class="grid-row text-base-lightest">
        <div class="tablet:grid-col-2"></div>
        <div class="tablet:grid-col-10">
          A service of the
          <a class="padding-x-05 usa-link--external" href="https://science.gsfc.nasa.gov/astrophysics">Astrophysics Science Division</a>
          at <a class="padding-x-05 usa-link--external" href="https://www.nasa.gov/goddard/">NASA/GSFC</a>
        </div>
      </div>
    </div>
  </div>
</footer>

<script src="/assets/uswds/js/uswds.min.js"></script>
<script src="/includes/js/xaminbox.js"></script>
<script src="/includes/js/nav.js"></script>
</body>
</html>
