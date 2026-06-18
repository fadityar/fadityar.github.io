const header = document.querySelector(".site-header");

const updateHeader = () => {
  const scrolled = window.scrollY > 20;
  header.style.background = scrolled
    ? "rgba(16, 19, 15, 0.86)"
    : "linear-gradient(to bottom, rgba(16, 19, 15, 0.86), transparent)";
  header.style.backdropFilter = scrolled ? "blur(16px)" : "none";
};

window.addEventListener("scroll", updateHeader, { passive: true });
updateHeader();

const featuredVideo = document.querySelector("#featuredVideo");
const videoFeature = document.querySelector(".video-feature");
const videoItems = document.querySelectorAll(".video-item");

videoItems.forEach((item) => {
  item.addEventListener("click", () => {
    if (!featuredVideo) return;

    videoItems.forEach((entry) => entry.classList.remove("active"));
    item.classList.add("active");
    videoFeature?.classList.toggle("portrait", item.dataset.orientation === "portrait");
    videoFeature?.classList.toggle("landscape", item.dataset.orientation !== "portrait");
    featuredVideo.pause();
    featuredVideo.poster = item.dataset.poster;
    featuredVideo.querySelector("source").src = item.dataset.video;
    featuredVideo.load();
    featuredVideo.play().catch(() => {});
  });
});
