package com.boot.service;

import java.util.List;

import com.boot.entity.Files;

public interface FilesService {

	void save(Files f);

	List<Files> getListByCard(Integer cardId);
}
