module Jekyll
  class TagPageGenerator < Generator
    safe true

    def generate(site)
      tags = site.posts.docs.flat_map { |post| post.data['tags'] || [] }.to_set
      tags.each do |tag|
        p = TagPage.new(site, site.source, tag)
        site.pages << p
      end
    end
  end

  class TagPage < Page
    def initialize(site, base, tag)
      @site = site
      @base = base
      @dir  = File.join('tag', tag)
      @name = 'index.html'

      self.process(@name)
     
      self.read_yaml(File.join(base, '_layouts'), 'tag.html')
      self.data['type'] = "tag_index"
      self.data['tag'] = tag
      self.data['title'] = "#{tag}"
    end
  end
end

