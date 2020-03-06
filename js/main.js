var normal = document.getElementById("nav-menu");
var reverse = document.getElementById("nav-menu-left");

var icon = normal !== null ? normal : reverse;
var offsetY = window.pageYOffset;
var maxwid;

// Toggle the "menu-open" % "menu-opn-left" classes
function toggle() {
	  var navRight = document.getElementById("nav");
	  var navLeft = document.getElementById("nav-left");
	  var nav = navRight !== null ? navRight : navLeft;

	  var button = document.getElementById("menu");
	  var site = document.getElementById("wrap");

	  if (nav.className == "menu-open" || nav.className == "menu-open-left") {
	  	  nav.className = "";
	  	  button.className = "";
	  	  button.style.right = '0px';

				site.style.position = 'relative';
				site.style.top = '0px';
				if (window.innerWidth > 742) {
					site.style.left = '0px'
				} else {
					site.style.maxWidth = maxwid;
				}
				window.scrollTo({top: offsetY});
	  } else if (reverse !== null) {
	  	  nav.className += "menu-open-left";
	  	  button.className += "btn-close";
				site.style.color = "red";
	  } else {
	  	  nav.className += "menu-open";
	  	  button.className += "btn-close";
				button.style.right = '12px';

				if (window.innerWidth > 742) {
					wid = (window.innerWidth - 742) * 0.5;
					site.style.left = wid + 'px';
				} else {
					maxwid = site.style.maxWidth;
					site.style.maxWidth = window.innerWidth - 12 + 'px';
				}
				offsetY = window.pageYOffset;
				site.style.top = -offsetY + 'px';
				site.style.position = 'fixed';
	    }
	}

// Ensures backward compatibility with IE old versions
function menuClick() {
	if (document.addEventListener && icon !== null) {
		icon.addEventListener('click', toggle);
	} else if (document.attachEvent && icon !== null) {
		icon.attachEvent('onclick', toggle);
	} else {
		return;
	}
}

menuClick();
